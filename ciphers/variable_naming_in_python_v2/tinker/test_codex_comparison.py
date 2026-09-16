"""Codex comparison contracts with all inference and Modal calls mocked.

Partitions: repeated models sharing identical inputs versus mismatched prompts or
limits; complete versus missing cases; approval denied/granted; concurrent text-mode
Luna responses; source/comparison artifact separation; and repeat-run rejection.
Resume: request-only versus saved SDK answers, no-op completed runs, and
ambiguous/mismatched SDK cache. Real shared parsing/scoring is exercised. No notebook tests, live model-quality
claims, subscription calls, or Modal sandboxes are included.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ciphers.variable_naming_in_python_v2.data.codex_apps import InferenceResult, PreflightResult, SecretTask
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsResult
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig
from ciphers.variable_naming_in_python_v2.tinker import codex_evaluate, screening_evaluate
from ciphers.variable_naming_in_python_v2.tinker.screening_prepare import Message, PreparedRequest, RequestBody, RunConfig, artifact_path


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    relative = Path("comparison-test")
    directory = artifact_path(relative)
    directory.mkdir()
    config = RunConfig(
        secret=SecretTask(cipher=CipherConfig(special_variables={"index": ("i", "j")}, length_bits=4), message_bits="101"),
        models=("model/a", "model/b"),
        num_problems=2,
    )
    (directory / "config.json").write_text(config.model_dump_json())
    rows = [
        PreparedRequest(
            request_id=f"{problem_id}:{model}",
            problem_id=problem_id,
            body=RequestBody(model=model, messages=(Message(content=f"EXACT PROMPT {problem_id}"),), max_tokens=config.max_tokens),
        )
        for problem_id in (10, 20)
        for model in config.models
    ]
    (directory / "requests.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in rows))
    (directory / "grading_cases.json").write_text(json.dumps({str(i): {"inputs": ["PRIVATE"], "outputs": ["EXPECTED"], "fn_name": None} for i in (10, 20)}))
    return relative, directory, rows


def test_comparison_preserves_prompts_cipher_cases_and_deduplicates_models(source_run):
    relative, source, rows = source_run
    target = artifact_path(codex_evaluate.prepare_codex_comparison(relative))
    cloned = [PreparedRequest.model_validate_json(line) for line in (target / "requests.jsonl").read_text().splitlines()]
    assert [row.problem_id for row in cloned] == [10, 20]
    for old, new in zip(rows[::2], cloned, strict=True):
        assert new.body.messages == old.body.messages
        assert new.body.max_tokens == old.body.max_tokens
        assert new.body.model == "gpt-5.6-luna" and new.request_id != old.request_id
    original_config = RunConfig.model_validate_json((source / "config.json").read_text())
    cloned_config = RunConfig.model_validate_json((target / "config.json").read_text())
    assert cloned_config.secret == original_config.secret
    assert (target / "grading_cases.json").read_bytes() == (source / "grading_cases.json").read_bytes()
    assert json.loads((target / "estimate.json").read_text())["estimated_usd"] is None
    with pytest.raises(FileExistsError):
        codex_evaluate.prepare_codex_comparison(relative)


@pytest.mark.parametrize("change", ["prompt", "limit", "cases"])
def test_mismatched_inputs_rejected_before_creating_comparison(source_run, change):
    relative, directory, rows = source_run
    if change == "cases":
        (directory / "grading_cases.json").write_text("{}")
    else:
        body = rows[1].body
        if change == "prompt":
            body.messages = (Message(content="DIFFERENT"),)
        else:
            body.max_tokens += 1
        (directory / "requests.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in rows))
    with pytest.raises(ValueError):
        codex_evaluate.prepare_codex_comparison(relative)
    assert not (directory / "codex-luna").exists()


def test_codex_approval_guard_does_not_call_sdk(source_run, monkeypatch):
    relative, _, _ = source_run
    infer = AsyncMock()
    monkeypatch.setattr(codex_evaluate, "infer", infer)
    with pytest.raises(ValueError, match="approved=True"):
        codex_evaluate.run_codex_comparison(relative)
    infer.assert_not_called()


def test_luna_uses_exact_text_prompts_and_shared_parallel_grading(source_run, monkeypatch):
    relative, source, _ = source_run
    target = codex_evaluate.prepare_codex_comparison(relative)

    async def infer(prompt, config, *, response_format):
        assert config.model == "gpt-5.6-luna"
        assert response_format == "text"
        return SimpleNamespace(text='{"code":"pass"}', model_dump=lambda **kwargs: {"prompt": prompt, "model": config.model})

    mock_infer = AsyncMock(side_effect=infer)
    monkeypatch.setattr(codex_evaluate, "infer", mock_infer)
    monkeypatch.setattr(screening_evaluate, "evaluate_on_modal", lambda *args: ModalAppsResult(status="passed", num_tests=1, sandbox_id="mock"))
    result_dir = codex_evaluate.run_codex_comparison(relative, approved=True, num_workers=16)
    assert result_dir == target
    assert sorted(call.args[0] for call in mock_infer.call_args_list) == ["EXACT PROMPT 10", "EXACT PROMPT 20"]
    rows = screening_evaluate.summarize(target)
    assert rows[0]["complete"] and rows[0]["functional_pass_at_1"] == 1
    assert rows[0]["joint_pass_at_1"] == 0 and rows[0]["reported_cost_usd"] is None
    assert not (source / "responses.jsonl").exists()
    records = [json.loads(line) for line in (artifact_path(target) / "responses.jsonl").read_text().splitlines()]
    assert {row["response"]["codex"]["prompt"] for row in records} == {"EXACT PROMPT 10", "EXACT PROMPT 20"}
    with pytest.raises(FileExistsError):
        codex_evaluate.run_codex_comparison(relative, approved=True, num_workers=16)
    assert mock_infer.await_count == 2


def save_sdk_answer(prompt, config, name):
    """Write a realistic SDK checkpoint without calling the provider; return it."""
    directory = artifact_path(config.artifact_subdir) / name
    directory.mkdir(parents=True)
    preflight = PreflightResult(
        model=config.model,
        artifact_base_dir=directory.parent,
        config_overrides=(),
        usage_limit_id="codex",
        usage_windows=(),
        checked_at=datetime.now(timezone.utc),
    )
    answer = InferenceResult(
        text='{"code":"pass"}',
        code=None,
        output_error=None,
        prompt=prompt,
        turn_id=name,
        requested_model=config.model,
        artifact_dir=str(directory),
        config_overrides=(),
        preflight=preflight,
        item_types=("agentMessage",),
    )
    (directory / "request.json").write_text(json.dumps({"prompt": prompt, "config": config.model_dump(mode="json"), "response_format": "text"}))
    (directory / "answer.json").write_text(answer.model_dump_json())
    return answer


@pytest.mark.parametrize("saved_final_answer", [False, True])
def test_codex_interrupt_resume_recovers_sdk_cache_and_finishes_missing(source_run, monkeypatch, saved_final_answer):
    """A killed turn has either request-only or answer-on-disk but no JSONL record."""
    relative, _, _ = source_run
    target = codex_evaluate.prepare_codex_comparison(relative)
    count = 0

    async def infer(prompt, config, *, response_format):
        nonlocal count
        count += 1
        answer = save_sdk_answer(prompt, config, str(count))
        if count == 1:
            if not saved_final_answer:
                (Path(answer.artifact_dir) / "answer.json").unlink()
            raise KeyboardInterrupt()
        return answer

    mock_infer = AsyncMock(side_effect=infer)
    monkeypatch.setattr(codex_evaluate, "infer", mock_infer)
    monkeypatch.setattr(screening_evaluate, "evaluate_on_modal", lambda *args: ModalAppsResult(status="passed", num_tests=1, sandbox_id="mock"))
    with pytest.raises(KeyboardInterrupt):
        codex_evaluate.run_codex_comparison(relative, approved=True, num_workers=1)
    assert (artifact_path(target) / "responses.jsonl").read_text() == ""
    codex_evaluate.run_codex_comparison(relative, approved=True, resume=True, num_workers=16)
    assert mock_infer.await_count == (2 if saved_final_answer else 3)
    rows = [json.loads(line) for line in (artifact_path(target) / "responses.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and len({r["request_id"] for r in rows}) == 2
    if saved_final_answer:
        recovered = next(row for row in rows if row["request_id"].startswith("10:"))
        assert recovered["response"]["codex"]["turn_id"] == "1"
    mock_infer.reset_mock()
    codex_evaluate.run_codex_comparison(relative, approved=True, resume=True)
    mock_infer.assert_not_called()


@pytest.mark.parametrize("mismatch", ["prompt", "model", "settings", "duplicate", "tools"])
def test_codex_resume_rejects_ambiguous_or_changed_sdk_cache(source_run, monkeypatch, mismatch):
    relative, _, _ = source_run
    target = codex_evaluate.prepare_codex_comparison(relative)
    directory = artifact_path(target)
    config = codex_evaluate.CodexInferenceConfig(model="gpt-5.6-luna", timeout_s=300, artifact_subdir=target / "generation")
    answer = save_sdk_answer("EXACT PROMPT 10", config, "cached")
    answer_path = Path(answer.artifact_dir) / "answer.json"
    if mismatch == "duplicate":
        save_sdk_answer("EXACT PROMPT 10", config, "duplicate")
    elif mismatch == "settings":
        path = answer_path.with_name("request.json")
        request = json.loads(path.read_text())
        request["config"]["timeout_s"] = 400
        path.write_text(json.dumps(request))
    else:
        data = json.loads(answer_path.read_text())
        if mismatch == "prompt":
            data["prompt"] = "different"
        elif mismatch == "model":
            data["requested_model"] = "different"
        else:
            data["item_types"] = ["commandExecution"]
        answer_path.write_text(json.dumps(data))
    (directory / "responses.jsonl").touch()
    mock_infer = AsyncMock()
    monkeypatch.setattr(codex_evaluate, "infer", mock_infer)
    with pytest.raises(ValueError):
        codex_evaluate.run_codex_comparison(relative, approved=True, resume=True)
    mock_infer.assert_not_called()
    assert (directory / "responses.jsonl").read_text() == ""
