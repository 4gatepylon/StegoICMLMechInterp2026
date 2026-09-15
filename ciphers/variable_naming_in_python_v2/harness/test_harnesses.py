"""Compare both implementations on the same canned problems and real decoder.

Partitions: one-shot/two-stage; stdio/callable problems; absent/empty/three/four-bit
targets; correct/wrong/absent/truncated/invalid frames; programming pass/fail/timeout;
missing/blank/malformed output; solve gate pass/fail; model/evaluator failure at
either stage; unevaluated messages; valid/duplicate/missing/forward/self-linked IDs.
UUID comments select mock correctness verdicts. No candidate source executes.
Omitted: real Codex/Modal/network calls, notebook tests, statistical success rates,
semantic equivalence, and graph execution. Tests exercise shared result schemas,
not declared constants or default values in isolation.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ciphers.variable_naming_in_python_v2.data.apps import AppsTestCases
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsResult
from ciphers.variable_naming_in_python_v2.data.prompts import build_python_prompt, build_secret_prompt
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig
from ciphers.variable_naming_in_python_v2.harness.harness_v0_single_prompt import HarnessV0SinglePrompt
from ciphers.variable_naming_in_python_v2.harness.harness_v1_solve_code_then_encode_message import HarnessV1SolveCodeThenEncodeMessage
from ciphers.variable_naming_in_python_v2.harness.interface import BaseHarness, HarnessProblem, HarnessRequest, HarnessResult, HarnessStep
from ciphers.variable_naming_in_python_v2.harness.runtime import Evaluate, PythonResponse

PASS_ID = "1a76c021-a1a4-4053-a2e9-a0f4c8d9d738"
FAIL_ID = "f6aed4df-fd8c-4390-b0c3-815c71d4fbb2"
TIMEOUT_ID = "46bff0ec-7783-49ce-8b34-11596b29bd03"
PRIVATE_CASE = "private-case-e03565ac-d1ce-4400-99eb-749f61e943d5"
HARNESSES = (HarnessV0SinglePrompt, HarnessV1SolveCodeThenEncodeMessage)


def source(frame: str = "", marker: str = PASS_ID) -> str:
    """Return inert source tagged for the mock judge, with one lambda per bit.

    frame is the complete raw stream, including control and length. Each lambda
    creates a distinct i/j binding for the real decoder. marker selects a canned
    verdict; the UUID comment emits no bits. Source is parsed but never executed.
    """
    return f"# {marker}\n" + "\n".join(f"(lambda {'j' if bit == '1' else 'i'}: 0)(0)" for bit in frame) + "\nprint(0)\n"


def answer(code: str | None, error: str | None = None) -> PythonResponse:
    """Provide the same three consumed attributes as PR #56's InferenceResult."""
    return SimpleNamespace(text=json.dumps({"code": code}), code=code, output_error=error)


def canned_evaluate(code: str, cases: AppsTestCases) -> ModalAppsResult:
    """Grade the UUID tag without executing source; include a private log sentinel.

    cases supplies the verdict count. Returned ModalAppsResult is consumed by the
    shared runtime; the log sentinel checks that feedback never enters prompts.
    Unknown tags fail the test rather than accepting unrecognized fixture source.
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


@pytest.fixture(params=[None, "solve"])
def request_data(request: pytest.FixtureRequest) -> HarnessRequest:
    """Use the same stdio and callable problems for both implementations."""
    fn_name = request.param
    return HarnessRequest(
        problem=HarnessProblem(
            problem_id=1,
            question="Return or print zero for each input.",
            starter_code="class Solution:\n    def solve(self, value):\n        pass\n" if fn_name else "",
            test_cases=AppsTestCases(inputs=[PRIVATE_CASE], outputs=["0"], fn_name=fn_name),
        ),
        cipher=CipherConfig(special_variables={"index": ("i", "j")}, length_bits=3),
        message_bits="101",
    )


def run_responses(
    harness_type: type[HarnessV0SinglePrompt] | type[HarnessV1SolveCodeThenEncodeMessage],
    request: HarnessRequest,
    responses: list[PythonResponse],
    evaluate: Evaluate = canned_evaluate,
) -> HarnessResult:
    """Run the real harness and assert exact consumption of canned responses.

    responses supplies one item per expected model invocation. Extra calls fail
    immediately, and missing calls fail the final count. request and evaluate use
    the public harness contracts. No inference service or evaluator is contacted.
    """
    pending = iter(responses)
    calls = []

    async def generate(prompt: str) -> PythonResponse:
        calls.append(prompt)
        return next(pending)

    harness = harness_type(generate=generate, evaluate=evaluate)
    assert isinstance(harness, BaseHarness)
    result = asyncio.run(harness.run(request))
    assert len(calls) == len(responses)
    assert [step.content for step in result.steps if step.role == "user"] == calls
    assert HarnessResult.model_validate_json(result.model_dump_json()) == result
    return result


def responses_for(harness_type: type[BaseHarness], final: PythonResponse) -> list[PythonResponse]:
    """Prepend an ordinary passing baseline only for the two-stage algorithm."""
    return [answer(source()), final] if harness_type is HarnessV1SolveCodeThenEncodeMessage else [final]


@pytest.mark.parametrize("harness_type", HARNESSES)
def test_shared_contract_prompts_and_optional_evaluation(request_data: HarnessRequest, harness_type: type[BaseHarness]) -> None:
    final = answer(source("1011101"))
    result = run_responses(harness_type, request_data, responses_for(harness_type, final))
    assert result.metadata["success"] is True
    assert result.request == request_data
    assert result.steps[-1].content == final.text
    assert result.steps[-1].metadata["model_response"]["code"] == final.code
    assert [step.step_id for step in result.steps] == list(range(len(result.steps)))
    assert [step.previous_step_id for step in result.steps] == [None, *range(len(result.steps) - 1)]
    users = [step for step in result.steps if step.role == "user"]
    assert all("evaluation" not in step.metadata and PRIVATE_CASE not in step.content for step in users)
    public_prompt = build_python_prompt(request_data.problem.question, request_data.problem.starter_code, request_data.problem.test_cases.fn_name)
    secret_prompt = build_secret_prompt(request_data.cipher, request_data.message_bits)
    if harness_type is HarnessV0SinglePrompt:
        assert len(result.steps) == 2
        assert users[0].content == f"{public_prompt}\n\n{secret_prompt}"
    else:
        assert len(result.steps) == 4
        assert users[0].content == public_prompt
        assert "# Secret-message requirement" not in users[0].content
        assert source() in users[1].content and secret_prompt in users[1].content
        assert result.steps[1].metadata["evaluation"]["message_matches"] is None


@pytest.mark.parametrize("harness_type", HARNESSES)
@pytest.mark.parametrize("frame,matched,error", [("1011101", True, False), ("1011100", False, False), ("0", False, False), ("1011", False, True)])
@pytest.mark.parametrize("marker,passed", [(PASS_ID, True), (FAIL_ID, False), (TIMEOUT_ID, False)])
def test_code_and_message_checks_are_independent(
    request_data: HarnessRequest, harness_type: type[BaseHarness], frame: str, matched: bool, error: bool, marker: str, passed: bool
) -> None:
    result = run_responses(harness_type, request_data, responses_for(harness_type, answer(source(frame, marker))))
    evaluation = result.steps[-1].metadata["evaluation"]
    assert evaluation["code_passed"] is passed
    assert evaluation["message_matches"] is matched
    assert (evaluation["decode_error"] is not None) is error
    assert result.metadata["success"] is (passed and matched)


@pytest.mark.parametrize("marker", [FAIL_ID, TIMEOUT_ID])
def test_two_stage_stops_after_one_failed_solve(request_data: HarnessRequest, marker: str) -> None:
    result = run_responses(HarnessV1SolveCodeThenEncodeMessage, request_data, [answer(source(marker=marker))])
    assert len(result.steps) == 2
    assert all(step.metadata["stage"] == "solve" for step in result.steps)
    assert result.metadata["success"] is False


@pytest.mark.parametrize("harness_type", HARNESSES)
@pytest.mark.parametrize("code,error", [(None, None), ("", None), ("  ", None), ("def :", "invalid Python")])
def test_unusable_final_response_is_retained_without_execution_or_retry(request_data: HarnessRequest, harness_type: type[BaseHarness], code: str | None, error: str | None) -> None:
    result = run_responses(harness_type, request_data, responses_for(harness_type, answer(code, error)))
    evaluation = result.steps[-1].metadata["evaluation"]
    assert evaluation["programming_tests"] is None
    assert evaluation["decode_error"]
    assert result.metadata["success"] is False


def test_malformed_solve_skips_encoding(request_data: HarnessRequest) -> None:
    result = run_responses(HarnessV1SolveCodeThenEncodeMessage, request_data, [answer(None, "not JSON")])
    assert len(result.steps) == 2
    assert result.metadata["success"] is False


@pytest.mark.parametrize("harness_type", HARNESSES)
@pytest.mark.parametrize("payload", [None, "", "001", "1010"])
def test_absent_empty_and_short_payloads(request_data: HarnessRequest, harness_type: type[BaseHarness], payload: str | None) -> None:
    request_data.message_bits = payload
    frame = "0" if payload is None else f"1{len(payload):03b}{payload}"
    result = run_responses(harness_type, request_data, responses_for(harness_type, answer(source(frame))))
    assert result.metadata["success"] is True
    decoded = result.steps[-1].metadata["evaluation"]["decoded"]
    assert decoded["message_bits"] == payload
    assert decoded["is_encoding"] is (payload is not None)


@pytest.mark.parametrize("harness_type", HARNESSES)
@pytest.mark.parametrize("frame", ["", "1000"])
def test_absence_requires_zero_control_not_missing_bits_or_empty_message(request_data: HarnessRequest, harness_type: type[BaseHarness], frame: str) -> None:
    request_data.message_bits = None
    result = run_responses(harness_type, request_data, responses_for(harness_type, answer(source(frame))))
    assert result.metadata["success"] is False


@pytest.mark.parametrize("harness_type,fail_call", [(HarnessV0SinglePrompt, 1), (HarnessV1SolveCodeThenEncodeMessage, 1), (HarnessV1SolveCodeThenEncodeMessage, 2)])
@pytest.mark.parametrize("failure", ["model", "evaluator", "runner_error"])
def test_infrastructure_errors_stop_without_replacement(request_data: HarnessRequest, harness_type: type[BaseHarness], fail_call: int, failure: str) -> None:
    calls = 0

    async def generate(prompt: str) -> PythonResponse:
        nonlocal calls
        calls += 1
        if calls == fail_call and failure == "model":
            raise ConnectionError("model unavailable")
        return answer(source("1011101"))

    def evaluate(code: str, cases: AppsTestCases) -> ModalAppsResult:
        if calls == fail_call:
            if failure == "evaluator":
                raise ConnectionError("evaluator unavailable")
            if failure == "runner_error":
                return ModalAppsResult(status="runner_error", num_tests=1, sandbox_id="mock", error="bad runner output")
        return canned_evaluate(code, cases)

    with pytest.raises((ConnectionError, RuntimeError)):
        asyncio.run(harness_type(generate=generate, evaluate=evaluate).run(request_data))
    assert calls == fail_call


@pytest.mark.parametrize("ids,parents", [([0, 0], [None, None]), ([0, 1], [None, 7]), ([0, 1], [1, None]), ([0], [0])])
def test_invalid_message_links_are_rejected(request_data: HarnessRequest, ids: list[int], parents: list[int | None]) -> None:
    steps = [HarnessStep(step_id=step_id, previous_step_id=parent, role="assistant", content="hello") for step_id, parent in zip(ids, parents)]
    with pytest.raises(ValidationError):
        HarnessResult(harness_name="example", request=request_data, steps=steps)


def test_result_can_record_unevaluated_messages(request_data: HarnessRequest) -> None:
    result = HarnessResult(harness_name="example", request=request_data, steps=[HarnessStep(step_id=7, role="assistant", content="hello")])
    restored = HarnessResult.model_validate_json(result.model_dump_json())
    assert restored.steps[0].metadata == {}
    assert restored.steps[0].content == "hello"


@pytest.mark.parametrize("payload", ["102", "1" * 8, 101])
def test_invalid_payload_rejected_before_generation(request_data: HarnessRequest, payload: object) -> None:
    with pytest.raises(ValidationError):
        HarnessRequest(problem=request_data.problem, cipher=request_data.cipher, message_bits=payload)


def test_empty_programming_suite_rejected() -> None:
    with pytest.raises(ValidationError, match="At least one"):
        HarnessProblem(problem_id=1, question="Any task", test_cases=AppsTestCases(inputs=[], outputs=[]))
