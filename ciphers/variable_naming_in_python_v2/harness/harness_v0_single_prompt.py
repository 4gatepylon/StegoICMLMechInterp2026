"""One prompt containing both the programming task and message requirement."""

from typing import override

from ciphers.variable_naming_in_python_v2.harness.interface import BaseHarness, HarnessRequest, HarnessResult, HarnessStep
from ciphers.variable_naming_in_python_v2.harness.runtime import message_prompt, problem_prompt, record_attempt


class HarnessV0SinglePrompt(BaseHarness):
    @override
    async def run(self, request: HarnessRequest) -> HarnessResult:
        """Implement BaseHarness.run with one combined task/message call.

        Return its user/assistant messages and metadata['success'], requiring both
        programming correctness and message recovery. Failures are not retried.
        """
        steps: list[HarnessStep] = []
        prompt = f"{problem_prompt(request)}\n\n{message_prompt(request)}"
        _, success = await record_attempt(request, steps, prompt, "encode", self.inference_config, self.modal_config, check_message=True)
        return HarnessResult(harness_name="harness_v0_single_prompt", request=request, steps=steps, metadata={"success": success})
