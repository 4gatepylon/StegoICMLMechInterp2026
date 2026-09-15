"""Contract tests for inference, prompts, framing examples, and pass@k.

Partitions: call-based/stdio interfaces; public/private prompt data; zero through
three payload bits (including leading zeroes), one-/multi-bit alphabets; valid,
blank, malformed, and syntactically invalid responses; ChatGPT/API-key/missing
accounts; inherited MCPs; tool-event violations; failures and deadlines. pass@k
is compared with exhaustive subset enumeration for small populations.

SDK calls are mocked. These tests do not execute candidate programs, test
notebooks, measure model quality, or contact Codex/Modal. Notebook-only joint
scoring is inspected in the live demo, not unit-tested here.
"""

import asyncio
import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tomllib
from openai_codex.generated.v2_all import TurnStatus
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data import codex_apps
from ciphers.variable_naming_in_python_v2.data.codex_apps import AppsPromptProblem, CodexInferenceConfig, SecretTask, build_apps_prompt, infer, pass_at_k
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig, decode


@pytest.mark.parametrize("fn_name", [None, "double"])
def test_prompt_preserves_public_specification_and_native_interface(fn_name: str | None) -> None:
    problem = AppsPromptProblem(question="PUBLIC_EXAMPLE: 2 -> 4", starter_code="# PUBLIC_SCAFFOLD", fn_name=fn_name)
    prompt = build_apps_prompt(problem)
    assert "PUBLIC_EXAMPLE" in prompt and "PUBLIC_SCAFFOLD" in prompt
    assert ("standard input" in prompt) == (fn_name is None)
    assert ("evaluator calls double" in prompt) == (fn_name is not None)
    for hidden_field in ("solutions", "input_output"):
        with pytest.raises(ValidationError):
            AppsPromptProblem(**problem.model_dump(), **{hidden_field: "PRIVATE_DATA"})


@pytest.mark.parametrize("payload", ["", "0", "1", "00", "01", "10", "11", "000", "001", "010", "011", "100", "101", "110", "111"])
@pytest.mark.parametrize("names", [("i", "j"), ("a", "b", "c", "d")])
def test_prompt_examples_decode_with_the_supplied_alphabet(payload: str, names: tuple[str, ...]) -> None:
    secret = SecretTask(cipher=CipherConfig(special_variables={"group": names}, length_bits=2), message_bits=payload)
    prompt = build_apps_prompt(AppsPromptProblem(question="Return a number."), secret=secret)
    assert f"required frame is {secret.frame_bits}" in prompt
    examples = re.findall(r'Example encoding "([01]*)":\n```python\n(.*?)\n```', prompt, re.DOTALL)
    assert len(examples) == 2
    for expected, source in examples:
        decoded = decode(source, secret.cipher)
        assert decoded.is_encoding and decoded.length == len(expected)
        assert decoded.message_bits == expected


@pytest.mark.parametrize("payload,width", [("0000", 2), ("x", 2), ("1", 4)])
def test_secret_rejects_incompatible_framing(payload: str, width: int) -> None:
    with pytest.raises(ValidationError):
        SecretTask(cipher=CipherConfig(special_variables={"index": ("i", "j")}, length_bits=width), message_bits=payload)


def test_pass_at_k_matches_every_small_population_and_subset() -> None:
    for n in range(1, 9):
        for c in range(n + 1):
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(range(n), k))
                empirical = sum(any(index < c for index in subset) for subset in subsets) / len(subsets)
                assert pass_at_k(n, c, k) == pytest.approx(empirical, abs=1e-14)


@pytest.mark.parametrize("n,c,k", [(0, 0, 1), (3, -1, 1), (3, 4, 1), (3, 1, 0), (3, 1, 4), (True, 0, 1), (3, 1.5, 1)])
def test_invalid_pass_at_k_counts(n, c, k) -> None:
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_pass_at_k_retains_small_nonzero_probabilities() -> None:
    assert pass_at_k(10**18, 1, 1) == pytest.approx(1e-18, rel=1e-12, abs=0)
    assert pass_at_k(10**18, 10**18 - 1, 1) == 1.0


@pytest.fixture
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, AsyncMock, AsyncMock]:
    """Mock both SDK clients; return constructor, inference client, config client.

    The config client supplies a server with punctuation to exercise TOML quoting;
    tests never launch it or print its credential-bearing configuration.
    """
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    config_client = AsyncMock()
    config_client.__aenter__.return_value = config_client
    config_client.request.return_value = SimpleNamespace(config=SimpleNamespace(model_extra={"mcp_servers": {"example.server": {"env": {"TOKEN": "never-save"}}}}))
    monkeypatch.setattr(codex_apps, "AsyncCodexClient", Mock(return_value=config_client))
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.account.return_value = SimpleNamespace(account=SimpleNamespace(root=SimpleNamespace(type="chatgpt")))
    client.thread_start.return_value.run.return_value = SimpleNamespace(
        status=TurnStatus.completed,
        id="turn-test",
        final_response=json.dumps({"code": "raise RuntimeError('never execute locally')"}),
        items=[SimpleNamespace(root=SimpleNamespace(type="agentMessage"))],
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr(codex_apps, "AsyncCodex", constructor)
    return constructor, client, config_client


def test_infer_restricts_tools_and_preserves_source_and_artifacts(fake_codex: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "never-forward")
    monkeypatch.setenv("CODEX_API_KEY", "never-forward")
    constructor, client, config_client = fake_codex
    answer = asyncio.run(infer("Solve this problem", CodexInferenceConfig(model="test-model"), response_format="python"))
    assert answer.code == "raise RuntimeError('never execute locally')" and answer.output_error is None
    saved_text = (Path(answer.artifact_dir) / "answer.json").read_text()
    assert json.loads(saved_text)["code"] == answer.code
    assert "never-save" not in saved_text
    runtime = constructor.call_args.args[0]
    assert "OPENAI_API_KEY" not in runtime.env and "CODEX_API_KEY" not in runtime.env
    mcp_override = next(value for value in runtime.config_overrides if value.startswith("mcp_servers="))
    assert tomllib.loads(mcp_override) == {"mcp_servers": {"example.server": {"enabled": False}}}
    assert "features.shell_tool=false" in runtime.config_overrides
    assert "tools.view_image=false" in runtime.config_overrides
    settings = client.thread_start.call_args.kwargs
    assert settings["sandbox"].value == "read-only" and settings["approval_mode"].value == "deny_all"
    assert settings["ephemeral"] and settings["model"] == "test-model"
    assert client.thread_start.return_value.run.call_args.kwargs["output_schema"]["properties"]["code"]["type"] == "string"
    assert list(Path(runtime.cwd).iterdir()) == []
    config_client.__aexit__.assert_awaited_once()
    client.__aexit__.assert_awaited_once()


def test_basic_text_inference_does_not_require_python(fake_codex: tuple) -> None:
    _, client, _ = fake_codex
    client.thread_start.return_value.run.return_value.final_response = "Paris"
    result = asyncio.run(infer("Capital of France?", CodexInferenceConfig()))
    assert result.text == "Paris" and result.code is None and result.output_error is None
    assert client.thread_start.return_value.run.call_args.kwargs["output_schema"] is None


@pytest.mark.parametrize("raw", [None, "", "Not JSON", '{"code": ""}', '{"code": "def broken(:"}', '{"unexpected": "pass"}'])
def test_malformed_completed_answers_are_returned_and_saved_as_failures(fake_codex: tuple, raw: str | None) -> None:
    _, client, _ = fake_codex
    client.thread_start.return_value.run.return_value.final_response = raw
    result = asyncio.run(infer("Solve", CodexInferenceConfig(), response_format="python"))
    assert result.output_error
    assert (Path(result.artifact_dir) / "answer.json").exists()
    client.thread_start.return_value.run.assert_awaited_once()


@pytest.mark.parametrize("account_type", [None, "apiKey"])
def test_non_subscription_auth_cannot_generate(fake_codex: tuple, account_type: str | None) -> None:
    _, client, _ = fake_codex
    client.account.return_value = SimpleNamespace(account=None if account_type is None else SimpleNamespace(root=SimpleNamespace(type=account_type)))
    with pytest.raises(RuntimeError, match="Sign in to Codex with ChatGPT"):
        asyncio.run(infer("Solve", CodexInferenceConfig()))
    client.thread_start.assert_not_called()


def test_tool_event_stops_experiment_after_saving_evidence(fake_codex: tuple, tmp_path: Path) -> None:
    _, client, _ = fake_codex
    client.thread_start.return_value.run.return_value.items = [SimpleNamespace(root=SimpleNamespace(type="commandExecution"))]
    with pytest.raises(RuntimeError, match="Unexpected Codex turn items"):
        asyncio.run(infer("Solve", CodexInferenceConfig()))
    assert len(list(tmp_path.rglob("answer.json"))) == 1


def test_failed_turn_stops_experiment(fake_codex: tuple) -> None:
    _, client, _ = fake_codex
    client.thread_start.return_value.run.return_value.status = TurnStatus.failed
    with pytest.raises(RuntimeError, match="did not complete"):
        asyncio.run(infer("Solve", CodexInferenceConfig()))


def test_timeout_closes_sdk(fake_codex: tuple) -> None:
    _, client, _ = fake_codex

    async def stalled_generation(*args, **kwargs):
        await asyncio.sleep(10)

    client.thread_start.return_value.run.side_effect = stalled_generation
    with pytest.raises(TimeoutError):
        asyncio.run(infer("Solve", CodexInferenceConfig(timeout_s=1)))
    client.__aexit__.assert_awaited_once()


def test_missing_artifact_root_fails_before_sdk(fake_codex: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    constructor, _, _ = fake_codex
    monkeypatch.delenv("STEGO_ARTIFACTS_DIR")
    with pytest.raises(ValueError, match="STEGO_ARTIFACTS_DIR"):
        asyncio.run(infer("Solve", CodexInferenceConfig()))
    constructor.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"timeout_s": 0}, {"artifact_subdir": "../escape"}, {"artifact_subdir": "/escape"}, {"model": " "}])
def test_invalid_generation_settings(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        CodexInferenceConfig(**kwargs)
