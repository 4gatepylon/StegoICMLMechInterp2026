"""Test the two-stage loop with canned inference/evaluation and the real decoder.

Partitions: first/later/no passing solve; 1/multiple solve budgets and encoding
samples; correct/wrong/absent/truncated/invalid frames; missing/blank/malformed
model output; code pass/fail/timeout independently of message recovery; public
stdio/callable prompts; configuration rejection; model/evaluator infrastructure
failure at either stage. UUID comments identify canned candidate verdicts and
let the mock model recognize prior answers without executing generated source.
Omitted: actual Codex/Modal/network execution, model success rates, semantic
equivalence, steganalysis, and notebook tests.
"""

import asyncio
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data.apps import AppsTestCases
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsResult
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig
from ciphers.variable_naming_in_python_v2.harness.harness_v1_solve_code_then_encode_message import (
    HarnessProblem,
    HarnessV1SolveCodeThenEncodeMessageConfig,
    HarnessV1SolveCodeThenEncodeMessageResult,
    ModelResponse,
    harness_v1_solve_code_then_encode_message,
)

PASS_ID = "1a76c021-a1a4-4053-a2e9-a0f4c8d9d738"
FAIL_ID = "f6aed4df-fd8c-4390-b0c3-815c71d4fbb2"
TIMEOUT_ID = "46bff0ec-7783-49ce-8b34-11596b29bd03"
PRIVATE_CASE = "private-case-e03565ac-d1ce-4400-99eb-749f61e943d5"


def source(frame: str = "", marker: str = PASS_ID) -> str:
    """Return inert fixture source: UUID verdict tag plus one scope per bit.

    frame is the raw control/length/payload stream, not just a payload. marker
    selects the canned evaluator outcome. Each lambda parameter is a distinct
    lexical binding, allowing repeated i/j symbols. Comments contribute no bits;
    the real decoder parses this code, but these tests never execute it.
    """
    return f"# {marker}\n" + "\n".join(f"(lambda {'j' if bit == '1' else 'i'}: 0)(0)" for bit in frame) + "\nprint(0)\n"


def answer(code: str | None, error: str | None = None) -> ModelResponse:
    """Wrap fixture source as extracted code and raw text for the model contract."""
    return ModelResponse(text=code or "malformed answer", code=code, output_error=error)


def canned_evaluate(code: str, cases: AppsTestCases) -> ModalAppsResult:
    """Return a verdict selected by the first-line UUID; never execute code.

    code must start with a known fixture marker. cases determines the verdict
    count. Logs intentionally contain a private sentinel to test that feedback
    does not expose evaluator diagnostics. Returned sandbox_id is a mock label.
    """
    marker = code.splitlines()[0].removeprefix("# ")
    status = {PASS_ID: "passed", FAIL_ID: "failed", TIMEOUT_ID: "timeout"}[marker]
    return ModalAppsResult(
        status=status,
        num_tests=len(cases.inputs),
        passed_tests=len(cases.inputs) if status == "passed" else 0,
        raw_results=[status == "passed"] * len(cases.inputs),
        logs=PRIVATE_CASE,
        sandbox_id="mock-sandbox",
    )


@pytest.fixture
def problem() -> HarnessProblem:
    """Supply public task text and a private test sentinel for leakage checks."""
    return HarnessProblem(problem_id=1, question="Print zero.", test_cases=AppsTestCases(inputs=[PRIVATE_CASE], outputs=["0"]))


def settings(**changes: object) -> HarnessV1SolveCodeThenEncodeMessageConfig:
    """Create the common cipher; keyword fields select behavioral partitions."""
    return HarnessV1SolveCodeThenEncodeMessageConfig(cipher=CipherConfig(special_variables={"index": ("i", "j")}, length_bits=3), message_bits="101", **changes)


def run_responses(
    problem: HarnessProblem,
    responses: list[ModelResponse],
    config: HarnessV1SolveCodeThenEncodeMessageConfig | None = None,
    evaluate: Callable[[str, AppsTestCases], ModalAppsResult] = canned_evaluate,
) -> HarnessV1SolveCodeThenEncodeMessageResult:
    """Run canned responses in order and fail on extra or missing model calls.

    problem/config configure the real harness. responses is consumed exactly
    once each. evaluate follows the harness's synchronous callback contract.
    Return the complete real result for assertions, without any external calls.
    """
    pending = iter(responses)
    prompts = []

    async def generate(prompt: str) -> ModelResponse:
        prompts.append(prompt)
        return next(pending)

    result = asyncio.run(harness_v1_solve_code_then_encode_message(problem, config or settings(), generate=generate, evaluate=evaluate))
    assert len(prompts) == len(responses)
    return result


def test_mock_model_uses_previous_uuid_to_repair_then_encodes(problem: HarnessProblem) -> None:
    prompts = []

    async def generate(prompt: str) -> ModelResponse:
        prompts.append(prompt)
        if "Required frame bits:" in prompt:
            return answer(source("1011101"))
        if FAIL_ID in prompt:
            assert "Passed 0/1 tests" in prompt
            return answer(source())
        return answer(source(marker=FAIL_ID))

    result = asyncio.run(harness_v1_solve_code_then_encode_message(problem, settings(), generate=generate, evaluate=canned_evaluate))
    assert [attempt.passed for attempt in result.solve_attempts] == [False, True]
    assert result.encodings[0].success
    assert len(prompts) == 3
    assert all(PRIVATE_CASE not in prompt for prompt in prompts)
    assert all("Cipher configuration:" not in prompt and "Requested message bits:" not in prompt for prompt in prompts[:2])
    assert FAIL_ID not in prompts[2]
    assert source() in prompts[2]
    assert "Read input from stdin" in prompts[0]
    restored = HarnessV1SolveCodeThenEncodeMessageResult.model_validate_json(result.model_dump_json())
    assert restored == result


@pytest.mark.parametrize("budget", [1, 3])
def test_exhausted_solve_budget_skips_encoding(problem: HarnessProblem, budget: int) -> None:
    result = run_responses(problem, [answer(source(marker=FAIL_ID))] * budget, settings(max_solve_attempts=budget))
    assert len(result.solve_attempts) == budget
    assert result.encodings == []


@pytest.mark.parametrize("frame,matched,error", [("1011101", True, False), ("1011100", False, False), ("0", False, False), ("1011", False, True)])
@pytest.mark.parametrize("marker,code_passed", [(PASS_ID, True), (FAIL_ID, False), (TIMEOUT_ID, False)])
def test_encoding_checks_are_independent_and_never_retried(problem: HarnessProblem, frame: str, matched: bool, error: bool, marker: str, code_passed: bool) -> None:
    result = run_responses(problem, [answer(source()), answer(source(frame, marker))])
    encoding = result.encodings[0]
    assert encoding.attempt.passed is code_passed
    assert encoding.message_matches is matched
    assert (encoding.decode_error is not None) is error
    assert encoding.success is (matched and code_passed)


def test_repeated_encodings_receive_identical_prompt_and_keep_failures(problem: HarnessProblem) -> None:
    result = run_responses(problem, [answer(source()), answer(source("0")), answer(source("1011101"))], settings(encoding_samples=2))
    first, second = result.encodings
    assert first.attempt.prompt == second.attempt.prompt
    assert [first.success, second.success] == [False, True]
    assert source("0") not in second.attempt.prompt


@pytest.mark.parametrize("response", [answer(None), answer(""), answer("  "), answer("def :", "invalid Python"), answer(source(), "bad response format")])
def test_unusable_solve_output_consumes_attempt_without_evaluation(problem: HarnessProblem, response: ModelResponse) -> None:
    result = run_responses(problem, [response, answer(source()), answer(source("1011101"))])
    assert result.solve_attempts[0].evaluation is None
    assert result.encodings[0].success


@pytest.mark.parametrize("response", [answer(None), answer(""), answer("def :", "invalid Python")])
def test_unusable_encoding_output_is_retained_without_repair(problem: HarnessProblem, response: ModelResponse) -> None:
    encoding = run_responses(problem, [answer(source()), response]).encodings[0]
    assert encoding.attempt.evaluation is None
    assert encoding.decode_error
    assert not encoding.success


@pytest.mark.parametrize("payload", ["", "001", "1010"])
def test_payload_length_and_leading_zeroes_reach_real_decoder(problem: HarnessProblem, payload: str) -> None:
    config = HarnessV1SolveCodeThenEncodeMessageConfig(cipher=settings().cipher, message_bits=payload)
    frame = f"1{len(payload):03b}{payload}"
    result = run_responses(problem, [answer(source()), answer(source(frame))], config)
    assert result.encodings[0].success
    assert result.encodings[0].decoded.message_bits == payload
    assert f"Required frame bits: {frame}" in result.encodings[0].attempt.prompt


def test_callable_interface_and_starter_are_in_both_stages(problem: HarnessProblem) -> None:
    problem.test_cases.fn_name = "solve"
    problem.starter_code = "class Solution:\n    def solve(self, value):\n        pass\n"
    result = run_responses(problem, [answer(source()), answer(source("1011101"))])
    for prompt in [result.solve_attempts[0].prompt, result.encodings[0].attempt.prompt]:
        assert "Implement the callable 'solve'" in prompt
        assert problem.starter_code in prompt
        assert PRIVATE_CASE not in prompt


@pytest.mark.parametrize("stage", ["solve", "encode"])
@pytest.mark.parametrize("failure", ["model", "evaluator", "runner_error"])
def test_infrastructure_failure_aborts_without_replacement(problem: HarnessProblem, stage: str, failure: str) -> None:
    calls = 0

    async def generate(prompt: str) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == (1 if stage == "solve" else 2) and failure == "model":
            raise ConnectionError("model unavailable")
        return answer(source("1011101"))

    def evaluate(code: str, cases: AppsTestCases) -> ModalAppsResult:
        if calls == (1 if stage == "solve" else 2):
            if failure == "evaluator":
                raise ConnectionError("evaluator unavailable")
            if failure == "runner_error":
                return ModalAppsResult(status="runner_error", num_tests=1, sandbox_id="mock", error="invalid worker output")
        return canned_evaluate(code, cases)

    with pytest.raises((ConnectionError, RuntimeError)):
        asyncio.run(harness_v1_solve_code_then_encode_message(problem, settings(), generate=generate, evaluate=evaluate))
    assert calls == (1 if stage == "solve" else 2)


@pytest.mark.parametrize("changes", [{"max_solve_attempts": 0}, {"max_solve_attempts": True}, {"encoding_samples": 0}, {"unknown": 1}])
def test_invalid_budgets_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(**changes)


@pytest.mark.parametrize("payload", ["102", "1" * 8])
def test_invalid_or_unrepresentable_payload_rejected(payload: str) -> None:
    with pytest.raises(ValidationError):
        HarnessV1SolveCodeThenEncodeMessageConfig(cipher=settings().cipher, message_bits=payload)


def test_empty_cases_rejected_before_inference() -> None:
    with pytest.raises(ValidationError, match="At least one test"):
        HarnessProblem(problem_id=1, question="Anything", test_cases=AppsTestCases(inputs=[], outputs=[]))
