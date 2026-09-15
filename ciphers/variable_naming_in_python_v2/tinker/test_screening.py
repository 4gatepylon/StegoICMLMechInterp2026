"""Offline contracts for the small OpenRouter screening harness.

Partitions: 1/100/>100 problems; available/missing models; stdio/call interfaces;
public/private data; nonzero input/output/request prices; absent/denied approval;
JSON/fenced JSON/blank/invalid Python/truncated output; correct/wrong/absent/empty/
leading-zero/truncated frames crossed with functional pass/fail; HTTP/API/Modal
infrastructure failure; complete/partial results and known/missing billed cost.

All HTTP and Modal execution are mocked. Real prompts and static decoding are used.
Omissions: notebooks, live service integration, model quality, tokenizer accuracy,
and internal decoder/Modal behavior already covered by their own test suites.
"""

import io
import json
from unittest.mock import Mock

import pytest
from datasets import Dataset
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data.apps import AppsTestCases
from ciphers.variable_naming_in_python_v2.data.codex_apps import SecretTask
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsConfig, ModalAppsResult
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig
from ciphers.variable_naming_in_python_v2.tinker import evaluate, prepare
from ciphers.variable_naming_in_python_v2.tinker.prepare import CatalogModel, Message, PreparedRequest, Pricing, RequestBody, RunConfig


@pytest.fixture
def secret() -> SecretTask:
    return SecretTask(cipher=CipherConfig(special_variables={"index": ("i", "j")}, length_bits=2), message_bits="101")


@pytest.fixture
def prepared(tmp_path, monkeypatch, secret):
    """Prepare two real prompts against fake data/catalog; no network calls."""
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    catalog = {"test/model": CatalogModel(id="test/model", pricing=Pricing(prompt=0.000001, completion=0.000002, request=0.01))}
    monkeypatch.setattr(prepare, "fetch_catalog", lambda _: catalog)
    rows = [
        {
            "problem_id": i,
            "question": "PUBLIC_TASK",
            "starter_code": "# PUBLIC_SCAFFOLD",
            "solutions": ["PRIVATE_REFERENCE"],
            "input_output": {"inputs": ["PRIVATE_INPUT"], "outputs": ["PRIVATE_OUTPUT"], "fn_name": "solve" if i % 2 else None},
        }
        for i in range(101)
    ]
    monkeypatch.setattr(prepare, "load_apps", lambda _: Dataset.from_list(rows))
    config = RunConfig(secret=secret, models=("test/model", "missing/model"), num_problems=2)
    relative = prepare.prepare_run(config)
    return relative, prepare.artifact_path(relative), config, catalog


def test_preparation_saves_exact_public_requests_and_separate_grading(prepared):
    _, directory, config, _ = prepared
    rows = [PreparedRequest.model_validate_json(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert len({row.request_id for row in rows}) == 2
    for row in rows:
        prompt = row.body.messages[0].content
        assert "PUBLIC_TASK" in prompt and "PUBLIC_SCAFFOLD" in prompt
        assert "PRIVATE_" not in prompt
        assert config.secret.frame_bits in prompt
    cases = json.loads((directory / "grading_cases.json").read_text())
    assert all(value["inputs"] == ["PRIVATE_INPUT"] for value in cases.values())
    assert "PRIVATE_REFERENCE" not in "".join(file.read_text() for file in directory.iterdir())
    estimate = json.loads((directory / "estimate.json").read_text())
    assert estimate["models"][1]["available"] is False
    assert estimate["models"][1]["requests"] == 0
    assert not (directory / "responses.jsonl").exists()


@pytest.mark.parametrize("count", [1, 100])
def test_one_request_per_problem_and_model_at_allowed_bounds(prepared, count):
    _, _, config, _ = prepared
    config = RunConfig.model_validate(config.model_copy(update={"num_problems": count}).model_dump())
    directory = prepare.artifact_path(prepare.prepare_run(config))
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == count
    assert len({(row["problem_id"], row["body"]["model"]) for row in rows}) == count


@pytest.mark.parametrize("count", [0, 101])
def test_problem_cap_rejects_out_of_range(secret, count):
    with pytest.raises(ValidationError):
        RunConfig(secret=secret, num_problems=count)


def test_cost_arithmetic_uses_tokens_not_million_token_rates(secret):
    config = RunConfig(secret=secret, models=("test/model",), num_problems=1, max_tokens=100, estimated_output_tokens=20)
    row = PreparedRequest(request_id="r", problem_id=1, body=RequestBody(model="test/model", messages=(Message(content="abcdef"),), max_tokens=100))
    catalog = {"test/model": CatalogModel(id="test/model", pricing=Pricing(prompt=0.001, completion=0.002, request=0.1))}
    estimate = prepare.estimate_cost([row], config, catalog)[0]
    assert estimate.input_characters == 6
    assert estimate.input_tokens == 18
    assert estimate.estimated_usd == pytest.approx(0.158)
    assert estimate.limit_scenario_usd == pytest.approx(0.318)


def test_catalog_ignores_unrequested_router_price_sentinels(monkeypatch):
    payload = {
        "data": [
            {"id": "router/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
            {"id": "test/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        ]
    }
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()))
    catalog = prepare.fetch_catalog(("test/model", "missing/model"))
    assert list(catalog) == ["test/model"]
    assert catalog["test/model"].pricing.prompt == 0.000001
    with pytest.raises(ValidationError):
        prepare.fetch_catalog(("router/auto",))


def test_multiple_models_share_prompts_for_both_invocation_interfaces(prepared, monkeypatch):
    _, _, config, catalog = prepared
    catalog["second/model"] = CatalogModel(id="second/model", pricing=Pricing(prompt=0, completion=0))
    config = RunConfig(secret=config.secret, models=tuple(catalog), num_problems=100)
    directory = prepare.artifact_path(prepare.prepare_run(config))
    rows = [PreparedRequest.model_validate_json(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 200
    prompts = {}
    for row in rows:
        prompts.setdefault(row.problem_id, []).append(row.body.messages[0].content)
    assert all(len(group) == 2 and group[0] == group[1] for group in prompts.values())
    cases = json.loads((directory / "grading_cases.json").read_text())
    assert {value["fn_name"] for value in cases.values()} == {None, "solve"}
    for problem_id, group in prompts.items():
        assert ("evaluator calls `solve`" in group[0]) == (cases[str(problem_id)]["fn_name"] == "solve")


@pytest.mark.parametrize("kwargs", [{}, {"approved": False}, {"approved": 1}])
def test_approval_required_before_any_inference(prepared, monkeypatch, kwargs):
    relative, directory, _, _ = prepared
    send = Mock(side_effect=AssertionError("Must not send"))
    monkeypatch.setattr(evaluate, "send_request", send)
    with pytest.raises(ValueError, match="approved=True"):
        evaluate.run_prepared(relative, **kwargs)
    send.assert_not_called()
    assert not (directory / "responses.jsonl").exists()


def completion(code: str, finish_reason="stop") -> dict:
    return {
        "choices": [{"message": {"content": json.dumps({"code": code})}, "finish_reason": finish_reason}],
        "usage": {"cost": 0.01, "prompt_tokens": 100, "completion_tokens": 20},
    }


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize(
    "bits,target,expected",
    [
        ("111101", "101", True),
        ("111001", "101", False),
        ("0", "", False),
        ("100", "", True),
        ("111001", "001", True),
        ("1111", "101", False),
    ],
)
def test_independent_functional_and_message_predicates(prepared, monkeypatch, functional, bits, target, expected):
    _, directory, config, _ = prepared
    row = PreparedRequest.model_validate_json((directory / "requests.jsonl").read_text().splitlines()[0])
    code = "\n".join(f"(lambda {'j' if bit == '1' else 'i'}: 0)(0)" for bit in bits)
    verdict = ModalAppsResult(status="passed" if functional else "failed", num_tests=1, passed_tests=int(functional), sandbox_id="mock")
    monkeypatch.setattr(evaluate, "evaluate_on_modal", lambda *args: verdict)
    secret = SecretTask(cipher=config.secret.cipher, message_bits=target)
    result = evaluate.evaluate_candidate(row, evaluate.ChatResponse.model_validate(completion(code)), AppsTestCases(inputs=[""], outputs=[""]), secret, ModalAppsConfig())
    assert result.functional_success is functional
    assert result.message_success is expected
    assert result.joint_success is (functional and expected)


@pytest.mark.parametrize("content,finish", [("", "stop"), ('{"code":""}', "stop"), ('{"code":"def !"}', "stop"), ('{"code":"pass"}', "length")])
def test_bad_answers_count_as_failures_without_execution(prepared, monkeypatch, content, finish):
    _, directory, config, _ = prepared
    row = PreparedRequest.model_validate_json((directory / "requests.jsonl").read_text().splitlines()[0])
    grade = Mock(side_effect=AssertionError("Must not execute"))
    monkeypatch.setattr(evaluate, "evaluate_on_modal", grade)
    response = evaluate.ChatResponse.model_validate({"choices": [{"message": {"content": content}, "finish_reason": finish}]})
    result = evaluate.evaluate_candidate(row, response, AppsTestCases(inputs=[""], outputs=[""]), config.secret, ModalAppsConfig())
    assert result.output_error and not result.joint_success
    grade.assert_not_called()


def test_raw_response_saved_before_grading_and_no_second_run(prepared, monkeypatch):
    relative, directory, _, _ = prepared
    send = Mock(return_value=completion("pass"))
    monkeypatch.setattr(evaluate, "send_request", send)

    def grade(*args):
        assert (directory / "responses.jsonl").read_text().count('"request_id"') == send.call_count
        return ModalAppsResult(status="passed", num_tests=1, passed_tests=1, sandbox_id="mock")

    monkeypatch.setattr(evaluate, "evaluate_on_modal", grade)
    results = evaluate.run_prepared(relative, approved=True)
    assert len(results) == 2
    for call, line in zip(send.call_args_list, (directory / "requests.jsonl").read_text().splitlines(), strict=True):
        assert call.args[0] == PreparedRequest.model_validate_json(line)
    summary = evaluate.summarize(relative)[0]
    assert summary["complete"] and summary["functional_pass_at_1"] == 1
    assert summary["joint_pass_at_1"] == 0
    assert summary["reported_cost_usd"] == pytest.approx(0.02)
    with pytest.raises(FileExistsError):
        evaluate.run_prepared(relative, approved=True)
    assert send.call_count == 2


@pytest.mark.parametrize("failure", ["http", "api", "modal"])
def test_infrastructure_failures_preserve_partial_records_and_suppress_rates(prepared, monkeypatch, failure):
    relative, directory, _, _ = prepared
    good = completion("pass")
    bad = RuntimeError("HTTP unavailable") if failure == "http" else {"error": {"message": "Unavailable"}} if failure == "api" else good
    monkeypatch.setattr(evaluate, "send_request", Mock(side_effect=[good, bad]))
    passed = ModalAppsResult(status="passed", num_tests=1, passed_tests=1, sandbox_id="mock")
    broken = ModalAppsResult(status="runner_error", num_tests=1, sandbox_id="mock", error="Broken runner")
    monkeypatch.setattr(evaluate, "evaluate_on_modal", Mock(side_effect=[passed, broken]))
    with pytest.raises(RuntimeError):
        evaluate.run_prepared(relative, approved=True)
    assert len((directory / "results.jsonl").read_text().splitlines()) == 1
    assert (directory / "error.json").exists()
    summary = evaluate.summarize(relative)[0]
    assert summary["completed"] == 1 and not summary["complete"]
    assert summary["joint_pass_at_1"] is None


def test_fenced_json_and_missing_cost(prepared, monkeypatch):
    relative, _, _, _ = prepared
    response = completion("pass")
    content = response["choices"][0]["message"]["content"]
    response["choices"][0]["message"]["content"] = f"```json\n{content}\n```"
    response.pop("usage")
    monkeypatch.setattr(evaluate, "send_request", Mock(return_value=response))
    monkeypatch.setattr(evaluate, "evaluate_on_modal", Mock(return_value=ModalAppsResult(status="passed", num_tests=1, passed_tests=1, sandbox_id="mock")))
    assert all(row.code == "pass" for row in evaluate.run_prepared(relative, approved=True))
    assert evaluate.summarize(relative)[0]["reported_cost_usd"] is None
