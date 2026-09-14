"""Validate prompt/authentication boundaries without model calls or code execution.

Partitions cover stdio versus function-call tasks; public versus hidden content;
valid/blank/invalid code; ChatGPT/API-key/missing accounts; complete/failed turns;
timeouts; and artifact persistence before grading. SDK interactions are mocked.
These tests omit live model quality, subscription availability, Modal execution,
and notebooks. No supplied or generated program is executed by this test suite.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai_codex.generated.v2_all import TurnStatus
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data import codex_apps
from ciphers.variable_naming_in_python_v2.data.codex_apps import CodexAppsConfig, PythonAnswer, build_stdio_prompt, generate_stdio_solution


def problem_row() -> dict:
    """Return a loader-schema task with distinct public/reference/hidden sentinels."""
    return {
        "problem_id": 1,
        "question": "Read an integer from stdin and print its double. PUBLIC_EXAMPLE: 2 -> 4",
        "starter_code": "# PUBLIC_SCAFFOLD",
        "input_output": {"inputs": ["HIDDEN_INPUT"], "outputs": ["HIDDEN_OUTPUT"], "fn_name": None},
        "solutions": ["REFERENCE_SOLUTION"],
        "url": "https://example.com/PRIVATE_SOURCE",
    }


@pytest.mark.parametrize("serialized", [False, True])
def test_prompt_describes_stdio_and_excludes_ground_truth_and_hidden_tests(serialized: bool) -> None:
    row = problem_row()
    if serialized:
        row["input_output"] = json.dumps(row["input_output"])
    prompt = build_stdio_prompt(row)
    for public in ["PUBLIC_EXAMPLE", "PUBLIC_SCAFFOLD", "standard input", "standard output", "Python 3.10"]:
        assert public in prompt
    for hidden in ["HIDDEN_INPUT", "HIDDEN_OUTPUT", "REFERENCE_SOLUTION", "PRIVATE_SOURCE"]:
        assert hidden not in prompt


def test_function_call_task_is_not_misrepresented_as_stdio() -> None:
    row = problem_row()
    row["input_output"]["fn_name"] = "solve"
    with pytest.raises(ValueError, match="standard-input/output"):
        build_stdio_prompt(row)


@pytest.mark.parametrize("code", ["", " \n", "```python\npass\n```", "def broken(:"])
def test_invalid_generated_source_is_rejected_without_execution(code: str) -> None:
    with pytest.raises(ValidationError):
        PythonAnswer(code=code)


def test_valid_generated_source_is_parsed_but_not_executed() -> None:
    code = "raise RuntimeError('Do not execute this locally')"
    assert PythonAnswer(code=code).code == code


@pytest.fixture
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, AsyncMock]:
    """Return mocked SDK constructor/client with a subscription and structured answer.

    The fixture sets the artifact root to pytest's temporary directory and makes
    account/thread/run asynchronous calls configurable by each test. It never
    starts the SDK runtime; tests inspect constructor options and saved records.
    """
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.account.return_value = SimpleNamespace(account=SimpleNamespace(root=SimpleNamespace(type="chatgpt")))
    thread = AsyncMock()
    thread.run.return_value = SimpleNamespace(status=TurnStatus.completed, id="turn-test", final_response=json.dumps({"code": "print(int(input()) * 2)"}))
    client.thread_start.return_value = thread
    constructor = Mock(return_value=client)
    monkeypatch.setattr(codex_apps, "AsyncCodex", constructor)
    return constructor, client


def test_generation_uses_subscription_and_saves_answer(fake_codex: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-forwarded")
    monkeypatch.setenv("CODEX_API_KEY", "test-key-never-forwarded")
    constructor, client = fake_codex
    answer = asyncio.run(generate_stdio_solution(problem_row(), CodexAppsConfig(model="test-model")))
    saved = json.loads((Path(answer.artifact_dir) / "answer.json").read_text())
    assert saved["code"] == "print(int(input()) * 2)"
    assert saved["turn_id"] == "turn-test"
    assert saved["requested_model"] == "test-model"
    assert "HIDDEN_INPUT" not in saved["prompt"]
    runtime = constructor.call_args.args[0]
    assert "OPENAI_API_KEY" not in runtime.env
    assert "CODEX_API_KEY" not in runtime.env
    assert "features.shell_tool=false" in runtime.config_overrides
    assert client.thread_start.call_args.kwargs["sandbox"].value == "read-only"
    assert client.thread_start.call_args.kwargs["approval_mode"].value == "deny_all"
    assert client.thread_start.call_args.kwargs["model_provider"] == "openai"
    assert list(Path(runtime.cwd).iterdir()) == []


@pytest.mark.parametrize("account_type", [None, "apiKey"])
def test_non_subscription_auth_cannot_start_a_turn(fake_codex: tuple, account_type: str | None) -> None:
    _, client = fake_codex
    account = None if account_type is None else SimpleNamespace(root=SimpleNamespace(type=account_type))
    client.account.return_value = SimpleNamespace(account=account)
    with pytest.raises(RuntimeError, match="Sign in to Codex with ChatGPT"):
        asyncio.run(generate_stdio_solution(problem_row()))
    client.thread_start.assert_not_called()
    client.__aexit__.assert_awaited_once()


def test_failed_turn_is_not_a_solution(fake_codex: tuple) -> None:
    _, client = fake_codex
    client.thread_start.return_value.run.return_value = SimpleNamespace(status=TurnStatus.failed, final_response=None)
    with pytest.raises(RuntimeError, match="did not complete"):
        asyncio.run(generate_stdio_solution(problem_row()))


def test_deadline_closes_sdk(fake_codex: tuple) -> None:
    _, client = fake_codex

    async def stalled_generation(*args, **kwargs):
        await asyncio.sleep(10)

    client.thread_start.return_value.run.side_effect = stalled_generation
    with pytest.raises(TimeoutError):
        asyncio.run(generate_stdio_solution(problem_row(), CodexAppsConfig(timeout_s=1)))
    client.__aexit__.assert_awaited_once()


def test_missing_artifact_root_fails_before_sdk(fake_codex: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    constructor, _ = fake_codex
    monkeypatch.delenv("STEGO_ARTIFACTS_DIR")
    with pytest.raises(ValueError, match="STEGO_ARTIFACTS_DIR"):
        asyncio.run(generate_stdio_solution(problem_row()))
    constructor.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"timeout_s": 0}, {"artifact_subdir": "../escape"}, {"artifact_subdir": "/escape"}, {"model": " "}])
def test_invalid_generation_settings(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        CodexAppsConfig(**kwargs)
