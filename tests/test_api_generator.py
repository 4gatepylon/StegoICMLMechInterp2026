"""Regression tests for the API generator's local request/response handling.

Partitions: empty/string/chat inputs; full/partial batches; positive/nonpositive
batch sizes; fresh/reused request options; successful/per-request/batch failures;
and object/non-object/invalid JSON with default/custom fallbacks. LiteLLM batch
calls are mocked, but successful responses use its real ModelResponse class.
Live providers, network retries, token streaming, and notebooks are omitted.
"""

from unittest.mock import Mock

import litellm
import pytest

from lib.utils.api_generator import APIGenerator


@pytest.fixture
def batch_completion(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the network boundary; tests configure results and inspect calls."""
    mock = Mock()
    monkeypatch.setattr(litellm, "batch_completion", mock)
    return mock


def response(text: str | None) -> litellm.ModelResponse:
    """Build a real SDK response whose first assistant message contains text."""
    return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": text}}])


@pytest.mark.parametrize("json_mode", [False, True])
def test_empty_inputs_make_no_request(batch_completion: Mock, json_mode: bool) -> None:
    generator = APIGenerator()
    generate = generator.api_generate_json_mode_streaming if json_mode else generator.api_generate_streaming
    assert list(generate([], "gpt-4o")) == []
    batch_completion.assert_not_called()


@pytest.mark.parametrize("prompt", ["hello", ""])
def test_single_string_is_one_user_message(batch_completion: Mock, prompt: str) -> None:
    batch_completion.return_value = [response("answer")]
    assert list(APIGenerator().api_generate_streaming(prompt, "gpt-4o")) == ["answer"]
    assert batch_completion.call_args.kwargs["messages"] == [[{"role": "user", "content": prompt}]]


@pytest.mark.parametrize("chat", [False, True])
def test_batching_preserves_input_and_output_order(batch_completion: Mock, chat: bool) -> None:
    prompts = ["one", "two", "three"]
    conversations = [[{"role": "system", "content": "Judge"}, {"role": "user", "content": text}] for text in prompts]
    batch_completion.side_effect = [[response("first"), response("second")], [response("third")]]
    results = list(APIGenerator().api_generate_streaming(conversations if chat else prompts, "gpt-4o", batch_size=2))
    assert results == ["first", "second", "third"]
    expected = conversations if chat else [[{"role": "user", "content": text}] for text in prompts]
    assert [call.kwargs["messages"] for call in batch_completion.call_args_list] == [expected[:2], expected[2:]]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_fails_before_request(batch_completion: Mock, batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        list(APIGenerator().api_generate_streaming(["hello"], "gpt-4o", batch_size=batch_size))
    batch_completion.assert_not_called()


def test_legacy_token_limit_does_not_leak_between_calls(batch_completion: Mock) -> None:
    batch_completion.return_value = [response("answer")]
    generator = APIGenerator()
    for limit in [8, 16, None]:
        assert list(generator.api_generate_streaming("hello", "gpt-4o", max_new_tokens=limit)) == ["answer"]
    options = [call.kwargs for call in batch_completion.call_args_list]
    assert [options[0]["max_tokens"], options[1]["max_tokens"]] == [8, 16]
    assert "max_tokens" not in options[2]


@pytest.mark.parametrize("legacy_limit", [None, 8])
def test_gpt5_token_translation_preserves_caller_options(batch_completion: Mock, legacy_limit: int | None) -> None:
    batch_completion.return_value = [response("answer")]
    options = {"temperature": 1} if legacy_limit else {"max_tokens": 8, "temperature": 1}
    original = options.copy()
    generator = APIGenerator()
    for model in ["gpt-5", "gpt-4o"]:
        list(generator.api_generate_streaming("hello", model, max_new_tokens=legacy_limit, batch_completion_kwargs=options))
    assert options == original
    first, second = [call.kwargs for call in batch_completion.call_args_list]
    assert first["max_completion_tokens"] == 8
    assert "max_tokens" not in first
    assert second["max_tokens"] == 8
    assert "max_completion_tokens" not in second


def test_conflicting_legacy_token_limit_is_rejected(batch_completion: Mock) -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        list(APIGenerator().api_generate_streaming("hello", "gpt-4o", max_new_tokens=8, batch_completion_kwargs={"max_tokens": 16}))
    batch_completion.assert_not_called()


@pytest.mark.parametrize("return_raw", [False, True])
def test_failures_preserve_positions_and_later_batches(batch_completion: Mock, return_raw: bool) -> None:
    success = response("answer")
    failure = RuntimeError("request failed")
    batch_completion.side_effect = [[success, failure], RuntimeError("batch failed"), [success]]
    results = list(APIGenerator().api_generate_streaming(["a", "b", "c", "d", "e"], "gpt-4o", batch_size=2, return_raw=return_raw))
    assert results == ([success, failure, None, None, success] if return_raw else ["answer", None, None, None, "answer"])


def test_failed_partial_batch_and_malformed_responses_yield_none(batch_completion: Mock) -> None:
    batch_completion.side_effect = [[object(), response(None)], RuntimeError("partial batch failed")]
    assert list(APIGenerator().api_generate_streaming(["a", "b", "c"], "gpt-4o", batch_size=2)) == [None, None, None]


def test_json_positional_arguments_reach_batch_completion(batch_completion: Mock) -> None:
    batch_completion.return_value = [response('{"score": 1}')]
    results = list(APIGenerator().api_generate_json_mode_streaming(["json", "json"], "gpt-4o", 2, 1, 32, False))
    assert results == [{"score": 1}, {"score": 1}]
    assert batch_completion.call_count == 2
    for call in batch_completion.call_args_list:
        assert call.kwargs["num_retries"] == 2
        assert call.kwargs["max_tokens"] == 32
        assert call.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("text", ["null", "42", "true", '"score"', '["score"]', "{}"])
@pytest.mark.parametrize("required_keys", [[], ["score"]])
def test_json_objects_and_required_keys(batch_completion: Mock, text: str, required_keys: list[str]) -> None:
    batch_completion.return_value = [response(text)]
    results = list(APIGenerator().api_generate_json_mode_streaming("json", "gpt-4o", must_have_keys=required_keys))
    assert results == ([{}] if text == "{}" and not required_keys else [{"error": "MissingKeys"}])


@pytest.mark.parametrize("custom_fallbacks", [False, True])
def test_json_failures_use_fallbacks_and_continue(batch_completion: Mock, custom_fallbacks: bool) -> None:
    batch_completion.return_value = [response(None), response("invalid"), response('{"other": 1}'), response('{"score": 2}')]
    options = {}
    if custom_fallbacks:
        options = {
            "default_json_for_none": {"unavailable": True},
            "default_json_for_keys_fn": lambda loaded: {"missing": loaded},
            "default_json_for_json_loads_decode_error_fn": lambda text, error: {"invalid": text, "position": error.pos},
        }
    results = list(APIGenerator().api_generate_json_mode_streaming(["json"] * 4, "gpt-4o", must_have_keys=["score"], **options))
    expected = (
        [{"unavailable": True}, {"invalid": "invalid", "position": 0}, {"missing": {"other": 1}}]
        if custom_fallbacks
        else [None, {"error": "JSONDecodeError"}, {"error": "MissingKeys"}]
    )
    assert results == [*expected, {"score": 2}]
