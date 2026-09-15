"""One prompt contains both the programming task and the message requirement."""

from typing import override

from ciphers.variable_naming_in_python_v2.data.modal_apps import evaluate_on_modal
from ciphers.variable_naming_in_python_v2.harness.interface import BaseHarness, HarnessRequest, HarnessResult, HarnessStep
from ciphers.variable_naming_in_python_v2.harness.runtime import Evaluate, Generate, message_prompt, problem_prompt, record_attempt


class HarnessV0SinglePrompt(BaseHarness):
    """Generate once and retain the answer regardless of correctness or decoding.

    Constructor callbacks generate/evaluate obey runtime.record_attempt's contract;
    evaluate defaults to the existing Modal runner. This composes PR #56's task
    and secret prompts in the same call. Repeat run() externally for more samples.
    Shared helper metadata is documented on record_attempt; result.metadata has
    only 'success', the boolean conjunction of code and message checks.
    """

    def __init__(self, *, generate: Generate, evaluate: Evaluate = evaluate_on_modal) -> None:
        self.generate = generate
        self.evaluate = evaluate

    @override
    async def run(self, request: HarnessRequest) -> HarnessResult:
        """Implement BaseHarness.run using one combined task/message generation.

        request supplies the task, cipher, and payload/absence target. Unlike the
        abstract base, this makes exactly one generation and checks the resulting
        code and message. Return a HarnessResult with a user and assistant message,
        including failures. Infrastructure exceptions propagate without retries.
        """
        steps: list[HarnessStep] = []
        prompt = f"{problem_prompt(request)}\n\n{message_prompt(request)}"
        _, success = await record_attempt(request, steps, prompt, "encode", self.generate, self.evaluate, check_message=True)
        return HarnessResult(harness_name="harness_v0_single_prompt", request=request, steps=steps, metadata={"success": success})
