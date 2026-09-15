"""Shared request and conversation record for the V2 Python harnesses."""

from abc import ABC, abstractmethod
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ciphers.variable_naming_in_python_v2.data.apps import AppsTestCases
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig


class HarnessProblem(BaseModel):
    """Public task/scaffold and private programming tests for one problem.

    problem_id identifies the APPS row (negative IDs can label local examples).
    question and starter_code go into prompts. test_cases contains paired inputs
    and outputs, plus fn_name: a callable entry point or None for stdin/stdout.
    Only fn_name is public; cases never enter prompts. Construct cases from loader
    rows with AppsTestCases.from_dataset_value(row['input_output']). References
    are not accepted. At least one test is required for the correctness gate.
    """

    model_config = ConfigDict(extra="forbid")
    problem_id: int
    question: str = Field(min_length=1)
    starter_code: str = ""
    test_cases: AppsTestCases

    @model_validator(mode="after")
    def require_tests(self) -> Self:
        """Reject an empty test suite before generating any candidates."""
        if not self.test_cases.inputs:
            raise ValueError("At least one programming test is required")
        return self


class HarnessRequest(BaseModel):
    """One task, cipher, and optional binary payload shared by every harness.

    message_bits=None requests absence with control zero; an empty string instead
    requests a present, zero-length message. Nonempty strings preserve leading
    zeroes. Payload length is derived and must fit cipher.length_bits. Algorithm
    dependencies belong in the harness constructor, not in this request.
    """

    model_config = ConfigDict(extra="forbid")
    problem: HarnessProblem
    
    cipher: CipherConfig
    # Cipher config includes finormation like:
    # ```
    # special_variables: dict[str, tuple[str, ...]]---a partitioning of the variable names into groups to encode bits
    # control_bits: Literal[1]---how many bits are used for "control coodes" (right now only 1 and it is a boolean code: do/don't encode)
    # length_bits: int = Field(default=4, ge=1, strict=True)---how many bits are used for the length of the message
    # ```
    message_bits: str | None = Field(default=None, pattern=r"^[01]*$", strict=True)
    # Message bits is the literal secret message to send

    @model_validator(mode="after")
    def require_representable_length(self) -> Self:
        """Reject payloads too long for the decoder's length header."""
        if self.message_bits is not None and len(self.message_bits).bit_length() > self.cipher.length_bits:
            raise ValueError("Message length does not fit the cipher's header")
        return self


class HarnessStep(BaseModel):
    """One conversation message and optional harness-specific JSON metadata.

    step_id is unique within its result. previous_step_id is None for a root or
    the ID of an earlier message. content records the exact prompt or raw model
    answer, not extracted code. metadata may be empty; absent evaluation metadata
    means no check is attached. Each implementation documents the keys it emits;
    consumers must not assume every harness evaluates every message.
    """

    model_config = ConfigDict(extra="forbid")
    step_id: int = Field(ge=0, strict=True)
    # TODO(hadriano): Steps may form a DAG when parallel trajectories merge; support multiple predecessors later.
    previous_step_id: int | None = Field(default=None, ge=0, strict=True)
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class HarnessResult(BaseModel):
    """A request and ordered message list returned by any BaseHarness.

    harness_name identifies the algorithm. metadata stores implementation-specific
    settings or summaries. Steps can omit evaluations entirely. The schema checks
    unique IDs and backward references; it does not interpret metadata, infer
    success, execute code, or implement graph traversal. Validate/serialize a
    completed result to retain its request, messages, and optional diagnostics.
    """

    model_config = ConfigDict(extra="forbid")
    harness_name: str = Field(min_length=1)
    request: HarnessRequest
    steps: list[HarnessStep] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_step_links(self) -> Self:
        """Reject duplicate IDs and missing, self, or forward predecessor links."""
        seen = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError("step_id must be unique within a result")
            if step.previous_step_id is not None and step.previous_step_id not in seen:
                raise ValueError("previous_step_id must identify an earlier step")
            seen.add(step.step_id)
        return self


class BaseHarness(ABC):
    """Common entry point; implementations supply their own generation strategy."""

    @abstractmethod
    async def run(self, request: HarnessRequest) -> HarnessResult:
        """Run one request and return its ordered conversation and metadata.

        request supplies the problem, cipher, and optional payload. Implementations
        must return HarnessResult, including ordinary candidate failures, and
        document their metadata schema. Infrastructure exceptions may propagate.
        This abstract method supplies no generation, evaluation, or retry policy.
        """
