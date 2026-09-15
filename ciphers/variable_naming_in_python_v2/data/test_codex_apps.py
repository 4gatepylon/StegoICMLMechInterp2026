"""Contract tests for inference, prompts, framing examples, and pass@k.

Partitions: call-based/stdio interfaces; public/private prompt data; zero through
three payload bits exhaustively, plus empty/leading-zero/capacity-boundary payloads
with 1/2/3/4/8-bit length fields; overflow and non-binary/non-string rejection;
one-/multi-bit active alphabets;
six fixed examples covering absent/empty/present/truncated frames, lexical scopes,
and bindings that span field boundaries (PyTorch source is parsed, never imported);
four active-cipher examples across payloads and group widths, exact/ignored bit
counts, and rejection of alphabets with no one-bit group;
valid, blank, malformed, and syntactically invalid responses; ChatGPT/API-key/missing
accounts; missing/unwritable artifacts; model typos, default resolution and catalog
pagination; quota boundaries, backend blocks and unavailable windows; inherited
MCPs; tool-event violations; failures and deadlines. pass@k
is compared with exhaustive subset enumeration for small populations.

SDK calls are mocked. These tests do not execute candidate programs, test
notebooks, measure model quality, or contact Codex/Modal. Notebook-only joint
scoring is inspected in the live demo, not unit-tested here.

TODO(hadriano) this file is not read by a human. The main testing was integration testing happening in `generate_apps_codex_modal.ipynb`
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
from openai_codex.generated.v2_all import GetAccountRateLimitsResponse, RateLimitSnapshot, RateLimitWindow, TurnStatus
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data import codex_apps
from ciphers.variable_naming_in_python_v2.data.codex_apps import (
    AppsPromptProblem,
    CodexInferenceConfig,
    CodexPreflightError,
    SecretTask,
    build_apps_prompt,
    infer,
    pass_at_k,
    preflight,
)
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig, IncompleteMessageError, decode


@pytest.mark.parametrize("fn_name", [None, "double"])
def test_prompt_preserves_public_specification_and_native_interface(fn_name: str | None) -> None:
    problem = AppsPromptProblem(question="PUBLIC_EXAMPLE: 2 -> 4", starter_code="# PUBLIC_SCAFFOLD", fn_name=fn_name)
    prompt = build_apps_prompt(problem)
    assert "PUBLIC_EXAMPLE" in prompt and "PUBLIC_SCAFFOLD" in prompt
    assert ("standard input" in prompt) == (fn_name is None)
    assert ("evaluator calls `double`" in prompt) == (fn_name is not None)
    for hidden_field in ("solutions", "input_output"):
        with pytest.raises(ValidationError):
            AppsPromptProblem(**problem.model_dump(), **{hidden_field: "PRIVATE_DATA"})


@pytest.mark.parametrize("payload", ["", "0", "1", "00", "01", "10", "11", "000", "001", "010", "011", "100", "101", "110", "111"])
@pytest.mark.parametrize("names", [("i", "j"), ("a", "b", "c", "d")])
def test_prompt_keeps_active_cipher_and_target_separate_from_example_ciphers(payload: str, names: tuple[str, ...]) -> None:
    secret = SecretTask(cipher=CipherConfig(special_variables={"group": names, "one_bit": ("zero", "one")}, length_bits=2), message_bits=payload)
    prompt = build_apps_prompt(AppsPromptProblem(question="Return a number."), secret=secret)
    assert f"Required frame: `{secret.frame_bits}`" in prompt
    active_alphabet = prompt.split("## Your active alphabet\n", 1)[1].split("## Binding and ordering rules", 1)[0]
    rows = re.findall(r"\| group \| `([^`]+)` \| `([01]+)` \|", active_alphabet)
    assert [name for name, _ in rows] == list(names)
    # The rendered mapping must agree with the decoder's actual emitted bits.
    for name, bits in rows:
        decoded = decode(f"{names[0]} = 0\ndef example({name}): pass", secret.cipher, keep_only_stego_bindings=True)
        assert decoded.bindings[-1].bits == bits
    assert "Example cipher A" not in active_alphabet and "Example cipher B" not in active_alphabet


@pytest.mark.parametrize(
    "example_index,expected_message,expected_bindings",
    [
        (0, None, [("i", "0", ("control",))]),
        (1, "0", [("j", "1", ("control",)), ("i", "0", ("length",)), ("j", "1", ("length",)), ("i", "0", ("message",))]),
        (3, "", [("j", "1", ("control",)), ("layer1", "00", ("length", "length")), ("i", "0", ("ignored",))]),
        (4, "01", [("c1", "11", ("control", "length")), ("layer1", "00", ("length", "message")), ("c2", "1", ("message",))]),
        (5, "010", [("c1", "11", ("control", "length")), ("conv_1", "10", ("length", "message")), ("c2", "1", ("message",)), ("conv_2", "0", ("message",))]),
    ],
    ids=["absent", "one-bit-payload", "present-empty", "two-bit-payload", "pytorch-three-bit-payload"],
)
def test_worked_examples_match_documented_frames_and_binding_roles(example_index, expected_message, expected_bindings) -> None:
    cipher = CipherConfig(
        special_variables={"index": ("i", "j")}
        if example_index < 3
        else {"index": ("i", "j"), "first_layer": ("layer1", "conv1", "conv_1", "c1"), "second_layer": ("conv_2", "c2")},
        length_bits=2,
    )
    prompt = build_apps_prompt(AppsPromptProblem(question="Return a number."), secret=SecretTask(cipher=cipher, message_bits="101"))
    examples = re.findall(r"```python\n(.*?)\n```", prompt, re.DOTALL)
    assert len(examples) == 10
    decoded = decode(examples[example_index], cipher, keep_only_stego_bindings=True)
    assert decoded.is_encoding == (expected_message is not None)
    assert decoded.length == (None if expected_message is None else len(expected_message))
    assert decoded.message_bits == expected_message
    assert [(binding.name, binding.bits, binding.bit_roles) for binding in decoded.bindings] == expected_bindings


def test_worked_truncated_example_has_one_of_three_declared_payload_bits() -> None:
    cipher = CipherConfig(special_variables={"index": ("i", "j")}, length_bits=2)
    prompt = build_apps_prompt(AppsPromptProblem(question="Return a number."), secret=SecretTask(cipher=cipher, message_bits="101"))
    source = re.findall(r"```python\n(.*?)\n```", prompt, re.DOTALL)[2]
    with pytest.raises(IncompleteMessageError, match="Truncated payload: declared 3 bits, found 1"):
        decode(source, cipher)


@pytest.mark.parametrize(
    "length_bits,payload",
    [(2, payload) for payload in ["", "0", "1", "00", "01", "10", "11", "000", "001", "010", "011", "100", "101", "110", "111"]]
    + [(length_bits, payload) for length_bits in [1, 3, 4, 8] for payload in ["", "0", "0" * (2**length_bits - 2) + "1"]]
    + [(3, "0010"), (4, "00101")],
)
@pytest.mark.parametrize("width", [1, 2, 3])
def test_active_examples_decode_exact_and_trailing_frames(payload: str, width: int, length_bits: int) -> None:
    cipher = CipherConfig(
        special_variables={"first": tuple(f"symbol_{index}" for index in range(2**width)), "one_bit": ("off", "on")},
        length_bits=length_bits,
    )
    prompt = build_apps_prompt(AppsPromptProblem(question="Return a number."), secret=SecretTask(cipher=cipher, message_bits=payload))
    # The six fixed examples from https://github.com/4gatepylon/StegoICMLMechInterp2026/pull/56 use separate ciphers; decode only the active section here.
    active_section = prompt.split("## Four examples using YOUR active cipher", 1)[1].split("## Apply this to your answer", 1)[0]
    examples = re.findall(r"```python\n(.*?)\n```", active_section, re.DOTALL)
    assert len(examples) == 4
    for index, (source, expected) in enumerate(zip(examples, [None, "", payload or "1", payload or "1"], strict=True)):
        decoded = decode(source, cipher, keep_only_stego_bindings=True)
        assert decoded.is_encoding == (expected is not None)
        assert decoded.length == (None if expected is None else len(expected))
        assert decoded.message_bits == expected
        roles = [role for binding in decoded.bindings for role in binding.bit_roles]
        assert roles.count("control") == 1
        assert roles.count("length") == (0 if expected is None else length_bits)
        assert roles.count("message") == len(expected or "")
        if index == 2:
            assert len(roles) == 1 + length_bits + len(expected)
            assert "ignored" not in roles
            assert all(len(binding.bits) == 1 for binding in decoded.bindings)
        if index == 3:
            assert roles.count("ignored") >= width
            assert roles[-width:] == ["ignored"] * width
        for binding_index, binding in enumerate(decoded.bindings, start=1):
            assert f"| {binding_index}: `{binding.name}` | `{binding.bits}` | {', '.join(binding.bit_roles)} |" in active_section


@pytest.mark.parametrize("width", [2, 3])
def test_secret_prompt_requires_one_bit_group_for_exact_example(width: int) -> None:
    cipher = CipherConfig(special_variables={"wide": tuple(f"symbol_{index}" for index in range(2**width))}, length_bits=2)
    secret = SecretTask(cipher=cipher, message_bits="101")
    with pytest.raises(ValueError, match="exactly two names.*Add a two-name group"):
        build_apps_prompt(AppsPromptProblem(question="Return a number."), secret=secret)


@pytest.mark.parametrize("length_bits", [1, 2, 3, 4, 8])
def test_secret_frame_round_trips_at_capacity_boundaries(length_bits: int) -> None:
    cipher = CipherConfig(special_variables={"index": ("i", "j")}, length_bits=length_bits)
    capacity = 2**length_bits - 1
    for payload_length in sorted({0, 1, capacity - 1, capacity}):
        payload = "" if payload_length == 0 else "0" * (payload_length - 1) + "1"
        secret = SecretTask(cipher=cipher, message_bits=payload)
        source = "\n".join(f"(lambda {('i', 'j')[int(bit)]}: None)(0)" for bit in secret.frame_bits)
        decoded = decode(source, cipher, keep_only_stego_bindings=True)
        assert decoded.is_encoding is True
        assert decoded.length == payload_length
        assert decoded.message_bits == payload
        assert [role for binding in decoded.bindings for role in binding.bit_roles] == (["control"] + ["length"] * length_bits + ["message"] * payload_length)
        prompt = build_apps_prompt(AppsPromptProblem(question="Return a number."), secret=secret)
        assert f"Required frame: `{secret.frame_bits}`" in prompt
        assert f"Length: exactly {length_bits} unsigned big-endian bits." in prompt
        assert f"Payload: 0 to {capacity} bits" in prompt


@pytest.mark.parametrize("length_bits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("overflow", [1, 2])
def test_secret_rejects_payload_lengths_beyond_capacity(length_bits: int, overflow: int) -> None:
    payload = "0" * (2**length_bits - 1 + overflow)
    with pytest.raises(ValidationError, match="Payload length exceeds the configured length field"):
        SecretTask(cipher=CipherConfig(special_variables={"index": ("i", "j")}, length_bits=length_bits), message_bits=payload)


@pytest.mark.parametrize("payload", ["x", "102", " 1", "1\n", 101, b"101", None])
def test_secret_rejects_non_binary_or_non_string_payloads(payload: object) -> None:
    with pytest.raises(ValidationError):
        SecretTask(cipher=CipherConfig(special_variables={"index": ("i", "j")}, length_bits=4), message_bits=payload)


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


@pytest.fixture
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, AsyncMock, AsyncMock]:
    """Mock both SDK clients; return constructor, inference client, config client.

    The config client supplies a server with punctuation to exercise TOML quoting;
    tests never launch it or print its credential-bearing configuration.
    """
    monkeypatch.setenv("STEGO_ARTIFACTS_DIR", str(tmp_path))
    config_client = AsyncMock()
    config_client.__aenter__.return_value = config_client
    config_client.account_read.return_value = SimpleNamespace(account=SimpleNamespace(root=SimpleNamespace(type="chatgpt")))
    config_client.effective = SimpleNamespace(config=SimpleNamespace(model=None, model_extra={"mcp_servers": {"example.server": {"env": {"TOKEN": "never-save"}}}}))
    config_client.model_pages = [
        SimpleNamespace(data=[SimpleNamespace(model="gpt-5.6-luna", is_default=True), SimpleNamespace(model="test-model", is_default=False)], next_cursor=None)
    ]
    config_client.usage = GetAccountRateLimitsResponse(
        ordinary_usage_allowed=True,
        rate_limits=RateLimitSnapshot(limit_id="codex", primary=RateLimitWindow(used_percent=20, window_duration_mins=10080, resets_at=1800000000)),
    )

    async def metadata_request(method, params, **kwargs):
        if method == "config/read":
            return config_client.effective
        if method == "model/list":
            return config_client.model_pages[int(params["cursor"] or 0)]
        if method == "account/rateLimits/read":
            return config_client.usage
        raise AssertionError(f"Unexpected RPC {method}")

    config_client.request.side_effect = metadata_request
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
    assert runtime.env["OPENAI_API_KEY"] == "" and runtime.env["CODEX_API_KEY"] == ""
    assert codex_apps.os.environ["OPENAI_API_KEY"] == "never-forward"
    assert codex_apps.os.environ["CODEX_API_KEY"] == "never-forward"
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
    _, client, config_client = fake_codex
    config_client.account_read.return_value = SimpleNamespace(account=None if account_type is None else SimpleNamespace(root=SimpleNamespace(type=account_type)))
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
    with pytest.raises(CodexPreflightError, match="STEGO_ARTIFACTS_DIR"):
        asyncio.run(infer("Solve", CodexInferenceConfig()))
    constructor.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"timeout_s": 0}, {"artifact_subdir": "../escape"}, {"artifact_subdir": "/escape"}, {"model": " "}])
def test_invalid_generation_settings(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        CodexInferenceConfig(**kwargs)


@pytest.mark.parametrize("model", ["gpt-5.5-luna", "does-not-exist"])
def test_bad_model_fails_with_suggestion_before_generation(fake_codex: tuple, model: str) -> None:
    constructor, _, _ = fake_codex
    with pytest.raises(CodexPreflightError) as caught:
        asyncio.run(infer("Solve", CodexInferenceConfig(model=model)))
    assert caught.value.stage == "model"
    assert model in str(caught.value) and "gpt-5.6-luna" in caught.value.hint
    constructor.assert_not_called()


def test_preflight_resolves_default_across_catalog_pages_without_generation(fake_codex: tuple) -> None:
    constructor, _, config_client = fake_codex
    config_client.effective.config.model = "second-page-model"
    config_client.model_pages[0].next_cursor = "1"
    config_client.model_pages.append(SimpleNamespace(data=[SimpleNamespace(model="second-page-model", is_default=False)], next_cursor=None))
    result = asyncio.run(preflight(CodexInferenceConfig(model=None)))
    assert result.model == "second-page-model"
    assert result.usage_windows[0].remaining_percent == 80
    assert list(result.artifact_base_dir.iterdir()) == []
    constructor.assert_not_called()


def test_invalid_configured_default_does_not_silently_fall_back(fake_codex: tuple) -> None:
    constructor, _, config_client = fake_codex
    config_client.effective.config.model = "old-invalid-model"
    with pytest.raises(CodexPreflightError, match="old-invalid-model"):
        asyncio.run(infer("Solve", CodexInferenceConfig(model=None)))
    constructor.assert_not_called()


def test_unwritable_artifact_root_has_remedy_and_original_cause(fake_codex: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    constructor, _, config_client = fake_codex
    monkeypatch.setattr(codex_apps, "TemporaryFile", Mock(side_effect=PermissionError("write denied")))
    with pytest.raises(CodexPreflightError) as caught:
        asyncio.run(infer("Solve", CodexInferenceConfig()))
    assert caught.value.stage == "artifacts"
    assert "writable" in caught.value.hint and isinstance(caught.value.__cause__, PermissionError)
    config_client.initialize.assert_not_called()
    constructor.assert_not_called()


@pytest.mark.parametrize("remaining, allowed", [(0, False), (9, False), (10, True), (11, True), (100, True)])
def test_reserve_threshold_checks_the_boundary_before_generation(fake_codex: tuple, remaining: int, allowed: bool) -> None:
    constructor, _, config_client = fake_codex
    config_client.usage.rate_limits.primary.used_percent = 100 - remaining
    if allowed:
        result = asyncio.run(preflight(CodexInferenceConfig(min_remaining_usage_percent=10)))
        assert result.usage_windows[0].remaining_percent == remaining
    else:
        with pytest.raises(CodexPreflightError) as caught:
            asyncio.run(infer("Solve", CodexInferenceConfig(min_remaining_usage_percent=10)))
        assert caught.value.stage == "usage"
        assert f"{remaining}% remaining" in str(caught.value) and "Reset:" in str(caught.value)
    constructor.assert_not_called()


def test_low_secondary_window_blocks_even_when_primary_is_healthy(fake_codex: tuple) -> None:
    constructor, _, config_client = fake_codex
    config_client.usage.rate_limits.secondary = RateLimitWindow(used_percent=99, window_duration_mins=300)
    with pytest.raises(CodexPreflightError, match="codex/secondary"):
        asyncio.run(infer("Solve", CodexInferenceConfig()))
    constructor.assert_not_called()


@pytest.mark.parametrize("failure", ["no_windows", "unknown_bucket", "usage_blocked", "spend_blocked", "limit_reached", "malformed_window"])
def test_unavailable_or_blocked_quota_cannot_start_generation(fake_codex: tuple, failure: str) -> None:
    constructor, _, config_client = fake_codex
    config = CodexInferenceConfig()
    if failure == "no_windows":
        config_client.usage.rate_limits.primary = None
    elif failure == "unknown_bucket":
        config = CodexInferenceConfig(usage_limit_id="missing")
    elif failure == "usage_blocked":
        config_client.usage.ordinary_usage_allowed = False
    elif failure == "spend_blocked":
        config_client.usage.rate_limits.spend_control_reached = True
    elif failure == "limit_reached":
        config_client.usage.rate_limits = RateLimitSnapshot(limit_id="codex", rate_limit_reached_type="rate_limit_reached")
    else:
        config_client.usage.rate_limits_by_limit_id = {"codex": {"primary": {"usedPercent": "bad"}}}
    with pytest.raises(CodexPreflightError) as caught:
        asyncio.run(infer("Solve", config))
    assert caught.value.stage == "usage" and caught.value.hint
    constructor.assert_not_called()


def test_quota_check_uses_selected_bucket_not_unrelated_exhausted_limits(fake_codex: tuple) -> None:
    _, _, config_client = fake_codex
    config_client.usage.rate_limits_by_limit_id = {
        "chosen": {"primary": {"usedPercent": 30, "windowDurationMins": 300}},
        "unrelated": {"primary": {"usedPercent": 100}},
    }
    result = asyncio.run(preflight(CodexInferenceConfig(usage_limit_id="chosen")))
    assert result.usage_limit_id == "chosen" and result.usage_windows[0].remaining_percent == 70


def test_explicitly_disabled_reserve_skips_usage_rpc_but_keeps_auth_and_model_checks(fake_codex: tuple) -> None:
    _, _, config_client = fake_codex
    result = asyncio.run(preflight(CodexInferenceConfig(min_remaining_usage_percent=None)))
    assert result.usage_windows == ()
    config_client.account_read.assert_awaited_once()
    methods = [call.args[0] for call in config_client.request.await_args_list]
    assert "model/list" in methods and "account/rateLimits/read" not in methods


def test_preflight_sdk_failure_has_stage_and_chained_cause(fake_codex: tuple) -> None:
    constructor, _, config_client = fake_codex
    config_client.initialize.side_effect = OSError("cannot start SDK")
    with pytest.raises(CodexPreflightError) as caught:
        asyncio.run(infer("Solve", CodexInferenceConfig()))
    assert caught.value.stage == "sdk" and isinstance(caught.value.__cause__, OSError)
    assert "openai-codex" in caught.value.hint
    config_client.__aexit__.assert_awaited_once()
    constructor.assert_not_called()


@pytest.mark.parametrize("threshold", [-1, 101, float("nan"), float("inf")])
def test_invalid_quota_policy_fails_at_configuration_construction(threshold: float) -> None:
    with pytest.raises(ValidationError):
        CodexInferenceConfig(min_remaining_usage_percent=threshold)


def test_infer_submits_the_resolved_default_and_records_it(fake_codex: tuple) -> None:
    _, client, config_client = fake_codex
    config_client.effective.config.model = "test-model"
    result = asyncio.run(infer("Solve", CodexInferenceConfig(model=None), response_format="python"))
    assert client.thread_start.call_args.kwargs["model"] == "test-model"
    assert result.requested_model is None and result.preflight.model == "test-model"
    saved = json.loads((Path(result.artifact_dir) / "request.json").read_text())
    assert saved["preflight"]["model"] == "test-model"


def test_standalone_preflight_deadline_preserves_the_failed_stage(fake_codex: tuple) -> None:
    constructor, _, config_client = fake_codex

    async def stalled_read(*args, **kwargs):
        await asyncio.sleep(10)

    config_client.account_read.side_effect = stalled_read
    with pytest.raises(CodexPreflightError) as caught:
        asyncio.run(preflight(CodexInferenceConfig(timeout_s=1)))
    assert caught.value.stage == "authentication"
    assert isinstance(caught.value.__cause__, TimeoutError)
    config_client.__aexit__.assert_awaited_once()
    constructor.assert_not_called()
