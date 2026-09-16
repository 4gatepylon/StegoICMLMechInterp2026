"""Test local Modal orchestration with mocked remote services and a fake evaluator.

Partitions: all-pass/mixed/negative/missing verdicts; normal/crashed/timed-out
processes; upload failure; empty submissions; stdio/function-call payloads;
faulthandler-compatible diagnostic capture and truncated logs on worker errors;
comment/blank-line versus code/string/docstring/indentation changes; fresh source
verification on each call and fail-closed behavior on mismatches/network errors.
No actual APPS solution runs locally. Remote credentials, image builds, and
upstream comparison correctness require a live Modal run and are omitted here.
No notebook tests are defined.
"""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data import modal_apps
from ciphers.variable_naming_in_python_v2.data.apps import AppsTestCases
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsConfig, evaluate_on_modal, interpret_verdict


def _uploaded_paths(fake_sandbox: Mock) -> dict[str, str]:
    """Map each Sandbox write destination to the uploaded text."""
    return {call.args[1]: call.args[0] for call in fake_sandbox.filesystem.write_text.call_args_list}


def _run_remote_driver(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Execute the worker script against a fake evaluator in ``tmp_path``."""
    environment = os.environ.copy()
    environment["STEGO_ARTIFACTS_DIR"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, str(modal_apps.REMOTE_DRIVER_PATH)],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("serialized", [False, True])
def test_dataset_cases_preserve_integers_beyond_machine_width(serialized: bool) -> None:
    payload = {"inputs": [[10**100]], "outputs": [10**101], "fn_name": "solve"}
    cases = AppsTestCases.from_dataset_value(json.dumps(payload) if serialized else payload)
    assert cases.inputs == [[10**100]]
    assert cases.outputs == [10**101]


@pytest.mark.parametrize(
    "results,num_tests,status,passed",
    [
        ([True, True], 2, "passed", 2),
        ([True, False], 2, "failed", 1),
        ([-1, -1], 2, "failed", 0),
        ([-2], 5, "failed", 0),
        ([], 2, "runner_error", 0),
        ([True], 2, "runner_error", 1),
        ([True, True], 1, "runner_error", 2),
    ],
)
def test_verdicts_require_literal_pass_for_every_case(results: list, num_tests: int, status: str, passed: int) -> None:
    result = interpret_verdict(json.dumps({"results": results, "logs": "", "error": None}), num_tests, "sb-test")
    assert (result.status, result.passed_tests) == (status, passed)
    assert result.raw_results == results


def test_worker_exception_prevents_pass() -> None:
    result = interpret_verdict(json.dumps({"results": [True], "logs": "details", "error": "failure"}), 1, "sb-test")
    assert result.status == "runner_error"
    assert result.logs == "details"


@pytest.mark.parametrize("stdout", ["not JSON", '{"results":[1],"logs":"","error":null}', '{"results":[true]}'])
def test_invalid_worker_messages_are_rejected(stdout: str) -> None:
    with pytest.raises(ValueError):
        interpret_verdict(stdout, 1, "sb-test")


@pytest.fixture
def fake_sandbox(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Return a mock Sandbox that accepts uploads and emits one successful verdict.

    The fixture replaces Modal lookup, image description, and Sandbox creation.
    Tests mutate process/file mocks to exercise lifecycle failures and inspect
    submissions without executing source or contacting Modal.
    """
    sandbox = Mock(object_id="sb-test")
    sandbox.exec.return_value.wait.return_value = 0
    sandbox.exec.return_value.stdout.read.return_value = json.dumps({"results": [True], "logs": "", "error": None})
    sandbox.exec.return_value.stderr.read.return_value = ""
    monkeypatch.setattr(modal_apps.modal.App, "lookup", Mock())
    monkeypatch.setattr(modal_apps.modal.Sandbox, "create", Mock(return_value=sandbox))
    monkeypatch.setattr(modal_apps, "evaluator_image", Mock())
    monkeypatch.setattr(modal_apps, "verified_evaluator_source", Mock(return_value="# verified evaluator\npass\n"))
    return sandbox


@pytest.mark.parametrize("fn_name,inputs", [(None, ["1\n"]), ("solve", [[{"a": 1}]])])
def test_full_cases_and_code_upload_without_local_execution(fake_sandbox: Mock, fn_name: str | None, inputs: list) -> None:
    code = "raise RuntimeError('must never run locally')"
    cases = AppsTestCases(inputs=inputs, outputs=[1], fn_name=fn_name)
    result = evaluate_on_modal(code, cases)
    uploads = _uploaded_paths(fake_sandbox)
    request_path = f"{modal_apps.REMOTE_ARTIFACTS_DIR}/request.json"
    driver_path = f"{modal_apps.REMOTE_ARTIFACTS_DIR}/{modal_apps.REMOTE_DRIVER_FILENAME}"
    payload = json.loads(uploads[request_path])
    assert payload["code"] == code
    assert payload["input_output"] == cases.model_dump()
    assert uploads[driver_path] == modal_apps.REMOTE_DRIVER_PATH.read_text()
    assert uploads[f"{modal_apps.REMOTE_ARTIFACTS_DIR}/apps_evaluator.py"] == modal_apps.verified_evaluator_source.return_value
    assert fake_sandbox.exec.call_args.args[:2] == ("python", driver_path)
    assert result.status == "passed"
    creation = modal_apps.modal.Sandbox.create.call_args.kwargs
    assert creation["block_network"] is True
    assert "secrets" not in creation
    fake_sandbox.terminate.assert_called_once()
    fake_sandbox.detach.assert_called_once()


@pytest.mark.parametrize("exit_code,status", [(-1, "timeout"), (137, "runner_error")])
def test_process_failure_cleans_up(fake_sandbox: Mock, exit_code: int, status: str) -> None:
    fake_sandbox.exec.return_value.wait.return_value = exit_code
    result = evaluate_on_modal("pass", AppsTestCases(inputs=[""], outputs=[""]))
    assert result.status == status
    fake_sandbox.terminate.assert_called_once()
    fake_sandbox.detach.assert_called_once()
    if exit_code == -1:
        fake_sandbox.exec.return_value.stdout.read.assert_not_called()


def test_upload_failure_still_terminates(fake_sandbox: Mock) -> None:
    fake_sandbox.filesystem.write_text.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        evaluate_on_modal("pass", AppsTestCases(inputs=[""], outputs=[""]))
    fake_sandbox.terminate.assert_called_once()
    fake_sandbox.detach.assert_called_once()


def test_malformed_result_becomes_runner_error(fake_sandbox: Mock) -> None:
    fake_sandbox.exec.return_value.stdout.read.return_value = "bad result"
    result = evaluate_on_modal("pass", AppsTestCases(inputs=[""], outputs=[""]))
    assert result.status == "runner_error"
    assert "Invalid evaluator result" in result.error
    fake_sandbox.terminate.assert_called_once()


@pytest.mark.parametrize("code,inputs", [(" ", [""]), ("pass", [])])
def test_empty_submissions_fail_before_modal(fake_sandbox: Mock, code: str, inputs: list) -> None:
    with pytest.raises(ValueError, match="nonblank"):
        evaluate_on_modal(code, AppsTestCases(inputs=inputs, outputs=inputs))
    modal_apps.modal.Sandbox.create.assert_not_called()
    modal_apps.verified_evaluator_source.assert_not_called()


@pytest.mark.parametrize(
    "original,annotated,equal",
    [
        ("x = 1\n", "# heading\n\nx = 1  # note\n", True),
        ("x=1\n", "x = 1\n", True),
        ("x = (1 +\n 2)\n", "x = (1 +  # note\n\n 2)\n", True),
        ("x = '# original'\n", "x = '# changed'\n", False),
        ('"""original docstring"""\nx = 1\n', '"""changed docstring"""\nx = 1\n', False),
        ("def f():\n    return 1\n", "def f():\n    return 2\n", False),
        ("if True:\n    x = 1\ny = 2\n", "if True:\n    x = 1\n    y = 2\n", False),
        ("x = 1\ny = 2\n", "x = 1; y = 2\n", False),
    ],
)
def test_source_comparison_ignores_only_comments_and_nonsemantic_spacing(original: str, annotated: str, equal: bool) -> None:
    assert (modal_apps._python_code_tokens(original) == modal_apps._python_code_tokens(annotated)) is equal


def test_verifier_reads_both_sources_afresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_path = tmp_path / "apps_evaluator.py"
    annotated = "# explanation\nx = '# literal'\n"
    local_path.write_text(annotated)
    monkeypatch.setattr(modal_apps, "EVALUATOR_PATH", local_path)
    download = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(b"x = '# literal'\n"))
    monkeypatch.setattr(modal_apps.urllib.request, "urlopen", download)
    assert modal_apps.verified_evaluator_source() == annotated
    local_path.write_text(annotated + "x += 1\n")
    with pytest.raises(AssertionError, match="differs from pinned upstream"):
        modal_apps.verified_evaluator_source()
    assert download.call_count == 2
    download.assert_called_with(modal_apps.EVALUATOR_URL, timeout=30)
    local_path.write_text(annotated)
    download.side_effect = lambda *args, **kwargs: io.BytesIO(b"x = '# changed upstream'\n")
    with pytest.raises(AssertionError, match="differs from pinned upstream"):
        modal_apps.verified_evaluator_source()
    download.side_effect = URLError("offline")
    with pytest.raises(URLError, match="offline"):
        modal_apps.verified_evaluator_source()


@pytest.mark.parametrize("failure", [AssertionError("source mismatch"), URLError("offline")])
def test_each_evaluation_verifies_before_contacting_modal(fake_sandbox: Mock, failure: Exception) -> None:
    cases = AppsTestCases(inputs=[""], outputs=[""])
    evaluate_on_modal("pass", cases)
    modal_apps.modal.App.lookup.reset_mock()
    modal_apps.modal.Sandbox.create.reset_mock()
    modal_apps.verified_evaluator_source.side_effect = failure
    with pytest.raises(type(failure)):
        evaluate_on_modal("pass", cases)
    assert modal_apps.verified_evaluator_source.call_count == 2
    modal_apps.modal.App.lookup.assert_not_called()
    modal_apps.modal.Sandbox.create.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"case_timeout_s": 0}, {"solution_timeout_s": -1}, {"memory_mb": 0}, {"case_timeout_s": 1.5}])
def test_invalid_resource_limits_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ModalAppsConfig(**kwargs)


def test_remote_driver_protocol_with_fake_evaluator(tmp_path: Path) -> None:
    # Only this hand-written fake runs locally; real upstream/APPS code stays remote.
    (tmp_path / "apps_evaluator.py").write_text(
        "import faulthandler\n"
        "def run_test(problem, test, debug):\n"
        "    faulthandler.enable()\n"
        "    faulthandler.dump_traceback()\n"
        "    faulthandler.disable()\n"
        "    print('fake diagnostic')\n"
        "    assert problem['input_output']['fn_name'] == 'solve'\n"
        "    assert test == 'not executable Python'\n"
        "    assert timeout == 7\n"
        "    return [True, False, -1]\n"
    )
    request = {"code": "not executable Python", "input_output": {"fn_name": "solve"}, "case_timeout_s": 7, "max_log_chars": 4000}
    (tmp_path / "request.json").write_text(json.dumps(request))
    process = _run_remote_driver(tmp_path)
    result = interpret_verdict(process.stdout, 3, "sb-fake")
    assert result.raw_results == [True, False, -1]
    assert "fake diagnostic" in result.logs
    assert "apps_evaluator.py" in result.logs


def test_remote_driver_preserves_error_and_truncates_unicode_logs(tmp_path: Path) -> None:
    (tmp_path / "apps_evaluator.py").write_text(
        "import sys\n"
        "def run_test(problem, test, debug):\n"
        "    print('discarded prefix')\n"
        "    print('retained café', file=sys.stderr)\n"
        "    raise RuntimeError('fake evaluator failure')\n"
    )
    request = {"code": "not executable Python", "input_output": {}, "case_timeout_s": 7, "max_log_chars": len("retained café\n")}
    (tmp_path / "request.json").write_text(json.dumps(request))
    process = _run_remote_driver(tmp_path)
    result = interpret_verdict(process.stdout, 1, "sb-fake")
    assert result.status == "runner_error"
    assert result.error.startswith("RuntimeError: fake evaluator failure")
    assert "Traceback (most recent call last)" in result.error
    assert result.logs.endswith("retained café\n")


@pytest.mark.parametrize("failure", [None, "verify", "upload"])
def test_profile_attributes_lifecycle_and_failures(fake_sandbox, tmp_path, monkeypatch, failure):
    """Separate known setup/wait/cleanup times, including early and upload failures."""
    from lib.utils import profiling

    clock = [0.0]
    monkeypatch.setattr(profiling, "perf_counter", lambda: clock[0])

    def step(seconds, value=None, error=False):
        clock[0] += seconds
        if error:
            raise RuntimeError("timed failure")
        return value

    modal_apps.verified_evaluator_source.side_effect = lambda: step(2, "pass", failure == "verify")
    modal_apps.modal.Sandbox.create.side_effect = lambda **kwargs: step(3, fake_sandbox)
    fake_sandbox.filesystem.write_text.side_effect = lambda *args: step(1, error=failure == "upload")
    fake_sandbox.exec.return_value.wait.side_effect = lambda: step(4, 0)
    fake_sandbox.terminate.side_effect = lambda: step(5)
    fake_sandbox.detach.side_effect = lambda: step(6)
    remote = {"read_request": 0.1, "import_evaluator": 0.2, "diagnostics": 0.1, "run_tests": 0.5, "total": 0.9}
    fake_sandbox.exec.return_value.stdout.read.return_value = json.dumps({"results": [True], "logs": "", "error": None, "timings_s": remote})
    path = tmp_path / "profile.jsonl"
    with profiling.Profiler(path).bind(model="model", request_id="request"):
        if failure:
            with pytest.raises(RuntimeError, match="timed failure"):
                evaluate_on_modal("pass", AppsTestCases(inputs=[""], outputs=[""]))
        else:
            result = evaluate_on_modal("pass", AppsTestCases(inputs=[""], outputs=[""]))
            assert result.remote_timings_s.run_tests == 0.5
    rows = {row.component: row for row in profiling.read_timings(path)}
    assert all(row.model == "model" and row.request_id == "request" for row in rows.values())
    assert rows["modal.verify_source"].elapsed_s == 2
    if failure == "verify":
        assert rows["modal.verify_source"].status == "error"
        assert "modal.sandbox_create" not in rows
    else:
        assert rows["modal.sandbox_create"].elapsed_s == 3
        assert rows["modal.terminate"].elapsed_s == 5
        assert rows["modal.detach"].elapsed_s == 6
        if failure == "upload":
            assert rows["modal.upload_request"].status == "error"
            assert "modal.process_wait" not in rows
        else:
            assert rows["modal.process_wait"].elapsed_s == 4
            assert rows["remote.run_tests"].elapsed_s == 0.5
            assert rows["remote.run_tests"].source == "remote"
