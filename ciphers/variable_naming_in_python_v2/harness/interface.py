"""Shared task and conversation schemas for the V2 harnesses."""

from abc import ABC, abstractmethod
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ciphers.variable_naming_in_python_v2.data.apps import AppsTestCases
from ciphers.variable_naming_in_python_v2.data.codex_apps import CodexInferenceConfig
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsConfig
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig


class HarnessProblem(BaseModel):
    """Public APPS ID/question/scaffold and private paired correctness tests.

    Only test_cases.fn_name enters prompts: None selects stdin/stdout; otherwise
    it names the callable. Load cases with AppsTestCases.from_dataset_value().
    """

    model_config = ConfigDict(extra="forbid")
    problem_id: int
    question: str = Field(min_length=1)
    starter_code: str = ""
    test_cases: AppsTestCases

    @model_validator(mode="after")
    def require_tests(self) -> Self:
        """Reject empty suites before spending inference calls."""
        if not self.test_cases.inputs:
            raise ValueError("At least one programming test is required")
        return self


class HarnessRequest(BaseModel):
    """Task and cipher; None requests control zero, while '' encodes an empty message."""

    model_config = ConfigDict(extra="forbid")
    problem: HarnessProblem
    cipher: CipherConfig
    message_bits: str | None = Field(default=None, pattern=r"^[01]*$", strict=True)

    @model_validator(mode="after")
    def require_representable_length(self) -> Self:
        """Reject payloads that cannot fit the configured length header."""
        if self.message_bits is not None and len(self.message_bits).bit_length() > self.cipher.length_bits:
            raise ValueError("Message length does not fit the cipher's header")
        return self


class HarnessStep(BaseModel):
    """Exact prompt/answer text, with optional metadata documented by its producer.

    The runtime assigns sequential IDs and links to the preceding message.
    This schema stores those links without implementing graph validation.
    """

    model_config = ConfigDict(extra="forbid")
    step_id: int
    # TODO(hadriano): Steps may form a DAG when parallel trajectories merge; support multiple predecessors later.
    previous_step_id: int | None
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class HarnessResult(BaseModel):
    """Original request and ordered messages; metadata holds harness-specific summaries."""

    model_config = ConfigDict(extra="forbid")
    harness_name: str
    request: HarnessRequest
    steps: list[HarnessStep]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class BaseHarness(ABC):
    """Shared configuration and entry point; each subclass implements its algorithm."""

    def __init__(self, inference_config: CodexInferenceConfig, modal_config: ModalAppsConfig) -> None:
        self.inference_config = inference_config
        self.modal_config = modal_config

    @abstractmethod
    async def run(self, request: HarnessRequest) -> HarnessResult:
        """Return the request's conversation, retaining candidate failures.

        Implementations document their metadata; infrastructure exceptions propagate.
        The constructor's existing Codex/Modal configs control model and resources.
        """
