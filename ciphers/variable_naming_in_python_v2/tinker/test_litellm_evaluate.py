"""Offline two-stage screening and profiling contracts.

Partitions: 1/10/100 prompts; one/multiple models; single/full/partial batches;
sequential models with real LiteLLM thread batching; grading 1/2/16 workers;
complete/partial/absent caches; request/batch/Modal failures and interruptions;
invalid settings/changed inputs; and known/absent cost and SDK timings.
Profiles use deterministic clocks for attribution and error semantics. Remote
services are mocked; notebooks, live speed/quality, and provider internals are
omitted. Existing suites cover APPS preparation, decoding and Codex scheduling.
"""

import json
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import Mock

import litellm
import pytest
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data.codex_apps import SecretTask
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsConfig, ModalAppsResult
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig
from ciphers.variable_naming_in_python_v2.tinker import litellm_evaluate as pipeline
from ciphers.variable_naming_in_python_v2.tinker import screening_evaluate as shared
from ciphers.variable_naming_in_python_v2.tinker.screening_prepare import Message, PreparedRequest, RequestBody, RunConfig, artifact_path
from lib.utils import profiling


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    """Create exact saved inputs with distinct prompts and private grading cases."""
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    index = 0

    def create(count=10, models=("test/a", "test/b")):
        nonlocal index
        index += 1
        relative = Path(f"run-{index}")
        directory = artifact_path(relative)
        directory.mkdir()
        config = RunConfig(num_problems=count, models=models, secret=SecretTask(cipher=CipherConfig(special_variables={"index": ("i", "j")}, length_bits=4), message_bits="101"))
        rows = [
            PreparedRequest(request_id=f"{i}:{model}", problem_id=i, body=RequestBody(model=model, messages=(Message(content=f"prompt {i}"),), max_tokens=config.max_tokens))
            for i in range(count)
            for model in models
        ]
        (directory / "config.json").write_text(config.model_dump_json())
        (directory / "requests.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in rows))
        (directory / "grading_cases.json").write_text(json.dumps({str(i): {"inputs": ["private"], "outputs": ["expected"]} for i in range(count)}))
        (directory / "estimate.json").write_text(json.dumps({"models": [{"model": model, "available": True, "requests": count} for model in models]}))
        return relative, directory, rows

    return create


def response(prompt="prompt", timing=1250):
    result = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": json.dumps({"code": f"# {prompt}\npass"})}, "finish_reason": "stop"}])
    result._response_ms = timing
    return result


@pytest.fixture
def batch_mock(monkeypatch):
    mock = Mock(side_effect=lambda **kwargs: [response(messages[0]["content"]) for messages in kwargs["messages"]])
    monkeypatch.setattr(litellm, "batch_completion", mock)
    return mock


@pytest.mark.parametrize("count,batch_size", [(1, 1), (10, 3), (100, 100)])
def test_batches_are_model_sequential_and_preserve_every_response(make_run, batch_mock, count, batch_size):
    relative, directory, requests = make_run(count)
    saved = pipeline.generate_prepared(relative, approved=True, batch_size=batch_size)
    assert [row.request_id for row in saved] == [row.request_id for row in requests]
    assert all(f"prompt {req.problem_id}" in saved_row.response["choices"][0]["message"]["content"] for req, saved_row in zip(requests, saved, strict=True))
    chunks = [min(batch_size, count - start) for start in range(0, count, batch_size)]
    calls = [call.kwargs for call in batch_mock.call_args_list]
    assert [call["model"] for call in calls] == [f"openrouter/{model}" for model in ("test/a", "test/b") for _ in chunks]
    assert [len(call["messages"]) for call in calls] == chunks * 2
    assert all(call["max_workers"] == len(call["messages"]) and call["num_retries"] == 0 for call in calls)
    assert all(call["max_tokens"] == requests[0].body.max_tokens and call["timeout"] == 300 for call in calls)
    assert not (directory / "results.jsonl").exists()
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    pipeline.generate_prepared(relative, approved=True, resume=True)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    times = profiling.read_timings(directory / "profile.jsonl")
    assert len([row for row in times if row.component == "generation.request"]) == 2 * count
    assert all(row.elapsed_s == 1.25 for row in times if row.component == "generation.request")


def test_real_litellm_batch_threads_overlap_within_one_model(make_run, monkeypatch):
    relative, _, _ = make_run(3)
    barrier = Barrier(3)
    lock = Lock()
    active_models = set()
    finished = []

    def completion(**kwargs):
        with lock:
            active_models.add(kwargs["model"])
            assert len(active_models) == 1
        barrier.wait(timeout=5)
        with lock:
            finished.append(kwargs["model"])
            if len(finished) % 3 == 0:
                active_models.clear()
        return response()

    monkeypatch.setattr(litellm, "completion", completion)
    pipeline.generate_prepared(relative, approved=True, batch_size=3)
    assert finished == ["openrouter/test/a"] * 3 + ["openrouter/test/b"] * 3


@pytest.mark.parametrize("workers", [1, 2, 16])
def test_all_generation_precedes_bounded_grading(make_run, batch_mock, monkeypatch, workers):
    relative, directory, requests = make_run(3)
    concurrency = min(workers, len(requests))
    barrier = Barrier(concurrency)
    lock = Lock()
    active = peak = started = 0

    def grade(*args):
        nonlocal active, peak, started
        assert len((directory / "responses.jsonl").read_text().splitlines()) == len(requests)
        assert batch_mock.call_count == 2
        with lock:
            first_wave = started < concurrency
            started += 1
            active += 1
            peak = max(peak, active)
        if first_wave:
            barrier.wait(timeout=5)
        with lock:
            active -= 1
        return ModalAppsResult(status="passed", num_tests=1, sandbox_id="fake")

    monkeypatch.setattr(shared, "evaluate_on_modal", grade)
    results = pipeline.run_prepared(relative, approved=True, num_workers=workers)
    assert peak == concurrency and active == 0
    assert [row.request_id for row in results] == [row.request_id for row in requests]
    times = profiling.read_timings(directory / "profile.jsonl")
    assert len([row for row in times if row.component == "grading.candidate"]) == len(requests)
    assert {row.request_id for row in times if row.component == "grading.persist"} == {row.request_id for row in requests}
    batch_mock.reset_mock()
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert pipeline.run_prepared(relative, approved=True, resume=True) == results
    batch_mock.assert_not_called()
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_out_of_order_grades_are_returned_in_input_order(make_run, batch_mock, monkeypatch):
    relative, directory, requests = make_run(3, ("test/a",))
    third_started = Event()

    def grade(code, *args):
        if "prompt 0" in code:
            assert third_started.wait(timeout=5)
        if "prompt 2" in code:
            third_started.set()
        return ModalAppsResult(status="failed", num_tests=1, sandbox_id="fake")

    monkeypatch.setattr(shared, "evaluate_on_modal", grade)
    results = pipeline.run_prepared(relative, approved=True, num_workers=2)
    assert json.loads((directory / "results.jsonl").read_text().splitlines()[0])["request_id"] == requests[1].request_id
    assert [row.request_id for row in results] == [row.request_id for row in requests]


@pytest.mark.parametrize("failure", ["request", "batch", "interrupt", "envelope", "count"])
def test_generation_failures_save_successes_and_resume_missing(make_run, batch_mock, monkeypatch, failure):
    relative, directory, requests = make_run(3)
    grade = Mock()
    monkeypatch.setattr(shared, "evaluate_on_modal", grade)
    error = KeyboardInterrupt if failure == "interrupt" else RuntimeError
    if failure in ("batch", "interrupt"):
        batch_mock.side_effect = error("unavailable")
    elif failure == "count":
        batch_mock.return_value = [response()]
        batch_mock.side_effect = None
    else:
        bad = RuntimeError("unavailable") if failure == "request" else litellm.ModelResponse(choices=[])
        batch_mock.side_effect = [[response("prompt 0"), bad, response("prompt 2")]]
    with pytest.raises(error):
        pipeline.run_prepared(relative, approved=True)
    grade.assert_not_called()
    assert batch_mock.call_count == 1
    cached = (directory / "responses.jsonl").read_text()
    expected_saved = 2 if failure in ("request", "envelope") else 0
    assert len(cached.splitlines()) == expected_saved
    batch_mock.side_effect = lambda **kwargs: [response(messages[0]["content"]) for messages in kwargs["messages"]]
    batch_mock.reset_mock()
    pipeline.generate_prepared(relative, approved=True, resume=True)
    assert sum(len(call.kwargs["messages"]) for call in batch_mock.call_args_list) == len(requests) - expected_saved
    assert (directory / "responses.jsonl").read_text().startswith(cached)
    assert (directory / "errors.jsonl").exists()
    assert any(row.status == ("interrupted" if failure == "interrupt" else "error") for row in profiling.read_timings(directory / "profile.jsonl"))


def test_grading_resume_never_needs_key_or_generation(make_run, batch_mock, monkeypatch):
    relative, directory, _ = make_run(3, ("test/a",))
    good = ModalAppsResult(status="failed", num_tests=1, sandbox_id="fake")
    grade = Mock(side_effect=[good, RuntimeError("Modal failed")])
    monkeypatch.setattr(shared, "evaluate_on_modal", grade)
    with pytest.raises(RuntimeError, match="Modal failed"):
        pipeline.run_prepared(relative, approved=True, num_workers=1)
    original = (directory / "results.jsonl").read_text()
    assert len(original.splitlines()) == 1
    monkeypatch.delenv("OPENROUTER_API_KEY")
    batch_mock.reset_mock()
    grade.side_effect = None
    grade.return_value = good
    results = pipeline.grade_prepared(relative, approved=True, num_workers=2, resume=True)
    assert len(results) == 3
    assert (directory / "results.jsonl").read_text().startswith(original)
    batch_mock.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": True}, {"batch_size": 1.5}, {"num_workers": 0}, {"num_workers": "16"}])
def test_invalid_settings_rejected_before_calls(make_run, batch_mock, kwargs):
    relative, directory, _ = make_run()
    with pytest.raises(ValidationError):
        pipeline.run_prepared(relative, approved=True, **kwargs)
    batch_mock.assert_not_called()
    assert not (directory / "responses.jsonl").exists()


def test_stage_preconditions_and_cache_corruption(make_run, batch_mock):
    relative, directory, _ = make_run()
    with pytest.raises(ValueError, match="approved"):
        pipeline.run_prepared(relative)
    with pytest.raises(FileNotFoundError):
        pipeline.generate_prepared(relative, approved=True, resume=True)
    with pytest.raises(ValueError, match="all responses"):
        pipeline.grade_prepared(relative, approved=True)
    pipeline.generate_prepared(relative, approved=True)
    with pytest.raises(FileExistsError):
        pipeline.generate_prepared(relative, approved=True)
    path = directory / "responses.jsonl"
    path.write_text(path.read_text() + '{"request_id":')
    batch_mock.reset_mock()
    with pytest.raises(ValueError, match="Incomplete"):
        pipeline.generate_prepared(relative, approved=True, resume=True)
    batch_mock.assert_not_called()


def test_changed_modal_settings_rejected_before_more_inference(make_run, batch_mock):
    relative, directory, _ = make_run()
    (directory / "responses.jsonl").touch()
    (directory / "execution.json").write_text(shared.ExecutionConfig().model_dump_json())
    with pytest.raises(ValueError, match="Modal settings"):
        pipeline.run_prepared(relative, approved=True, resume=True, modal_config=ModalAppsConfig(memory_mb=2048))
    batch_mock.assert_not_called()


def test_reported_cost_and_absent_latency_are_not_invented(make_run, batch_mock):
    relative, directory, _ = make_run(1, ("test/a",))
    raw = response(timing=None)
    raw._hidden_params["additional_headers"] = {"llm_provider-x-litellm-response-cost": "0.012"}
    batch_mock.side_effect = None
    batch_mock.return_value = [raw]
    pipeline.generate_prepared(relative, approved=True)
    assert pipeline.summarize(relative)[0]["reported_cost_usd"] == 0.012
    assert not any(row.component == "generation.request" for row in profiling.read_timings(directory / "profile.jsonl"))
    no_cost = response()
    assert pipeline.response_json(no_cost)["usage"].get("cost") is None


def test_profile_nested_time_errors_and_legacy_omissions(tmp_path, monkeypatch):
    ticks = iter([0, 1, 4, 5, 10, 12, 20, 23])
    monkeypatch.setattr(profiling, "perf_counter", lambda: next(ticks))
    path = tmp_path / "profile.jsonl"
    recorder = profiling.Profiler(path)
    with recorder.bind(model="model", request_id="request"):
        with profiling.measure("outer", items=2):
            with profiling.measure("inner", items=1):
                pass
        with pytest.raises(RuntimeError), profiling.measure("failed", items=1):
            raise RuntimeError("error")
        with pytest.raises(KeyboardInterrupt), profiling.measure("interrupted", items=1):
            raise KeyboardInterrupt()
    rows = {row.component: row for row in profiling.read_timings(path)}
    assert rows["outer"].elapsed_s == 5 and rows["inner"].elapsed_s == 3
    assert rows["failed"].status == "error" and rows["failed"].items == 0
    assert rows["interrupted"].status == "interrupted" and rows["interrupted"].elapsed_s == 3
    assert all(row.model == "model" and row.request_id == "request" for row in rows.values())
    summary = {row["component"]: row for row in profiling.summarize_timings(path)}
    assert summary["outer"]["items_per_s"] == 0.4
    assert summary["inner"]["median_s"] == summary["inner"]["p90_s"] == 3
    assert profiling.summarize_timings(tmp_path / "missing.jsonl") == []
    assert all(row["invocation_id"] == recorder.invocation_id for row in summary.values())


def test_profile_percentiles_resumes_and_corruption(tmp_path):
    path = tmp_path / "profile.jsonl"
    first = profiling.Profiler(path)
    for seconds in (1, 2, 4):
        with first.bind(model="model"):
            profiling.report_duration("generation.request", seconds, source="sdk", items=1)
    second = profiling.Profiler(path)
    with second.bind(model="model"):
        profiling.report_duration("generation.request", 10, source="sdk", items=1)
    rows = profiling.summarize_timings(path)
    assert len(rows) == 2
    assert rows[0]["total_s"] == 7 and rows[0]["median_s"] == 2
    assert rows[0]["p90_s"] == pytest.approx(3.6)
    assert rows[1]["total_s"] == 10
    path.write_text(path.read_text() + '{"partial":')
    with pytest.raises(ValueError, match="Incomplete"):
        profiling.Profiler(path)
