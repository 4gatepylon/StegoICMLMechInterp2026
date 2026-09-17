"""Version-one configuration and disk records for APPS trajectory generation.

These models are implemented; pipeline execution is not. Consumers should parse
JSON with model_validate_json and serialize with model_dump_json. Unknown fields
are rejected. Cross-file joins, source hashes, prompt correctness, and sampling
distributions must be checked by the future producer; individual models cannot
verify them. All stored paths are relative to STEGO_ARTIFACTS_DIR.
"""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from ciphers.variable_naming_in_python_v2.data.apps import AppsConfig, AppsTestCases
from ciphers.variable_naming_in_python_v2.data.codex_apps import AppsPromptProblem, CodexInferenceConfig
from ciphers.variable_naming_in_python_v2.data.modal_apps import ModalAppsConfig, ModalAppsResult
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig, DecodedMessage

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$", strict=True)]
Payload = Annotated[str, Field(pattern=r"^[01]+$", min_length=1, max_length=8, strict=True)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
PositiveInt = Annotated[int, Field(ge=1, strict=True)]
NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]
ControlBit = Annotated[int, Field(ge=0, le=1, strict=True)]
Stage = Literal["prepare", "generate", "grade"]
Condition = Literal["ordinary", "framed"]


class Record(BaseModel):
    """Reject unknown fields in configuration and artifact records.

    Subclasses document their entire stored schema, including inherited fields.
    Assignment is validated as well as construction; this does not make nested
    containers immutable. Producers should create new records for updates.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SamplingConfig(Record):
    """Configure independent request draws, separately from model inference.

    min_bits/max_bits: Inclusive payload-length bounds within 1..8.
    lengths_per_problem: Number of uniform length draws WITH replacement.
    bitstrings_per_length: Number of independently drawn payloads per length draw.
    ordinary_generations_per_problem: Additional independent generations under the
        ordinary APPS prompt, with no cipher/objective. Zero disables this baseline.
    payload_one_probability: Bernoulli probability of each payload bit being one.
    control_one_probability: Independent probability of requesting a present frame.
    seed: Local sampling seed; it does not seed Codex. Duplicate draws are retained.

    The future sampler draws a length, then each payload and its control bit,
    before drawing the next length. Lengths stay uniform when probabilities change.
    Endpoint probabilities 0 and 1 are supported for deterministic experiments.
    """

    min_bits: int = Field(default=1, ge=1, le=8, strict=True)
    max_bits: int = Field(default=8, ge=1, le=8, strict=True)
    lengths_per_problem: PositiveInt = 4
    bitstrings_per_length: PositiveInt = 4
    ordinary_generations_per_problem: NonnegativeInt = 16
    payload_one_probability: Probability = 0.5
    control_one_probability: Probability = 0.5
    seed: int = Field(default=42, strict=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Reject an empty inclusive length range; return the validated configuration."""
        if self.min_bits > self.max_bits:
            raise ValueError("min_bits must not exceed max_bits")
        return self


class RunConfig(Record):
    """Resolved, persisted settings shared by all stages of one run.

    sampling: Local request sampling settings.
    cipher: Alphabet/framing shared by prompts and decoding; normally length_bits=4.
        At least one two-name group is required by the existing prompt renderer.
    apps: Existing APPS split, revision, and eligibility filters. All eligible
        problems are used unless max_problems limits the ascending problem_id order.
    max_problems: None means all eligible problems; otherwise take the first N.
    inference: Existing Codex model, timeout, and quota settings. The future adapter
        replaces artifact_subdir with the run's sdk/ directory, using model_copy;
        manifest configuration must record that resolved relative directory.
    modal: Existing remote evaluator resource settings.
    generation_workers/grading_workers: Independent maximum in-flight candidates.

    Pydantic enforces field bounds and cipher capacity here. No services are called.
    The single CLI reads this JSON once during prepare and persists resolved values.
    Later stages use the manifest; configuration overrides on resume are rejected.
    """

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    cipher: CipherConfig
    apps: AppsConfig = Field(default_factory=AppsConfig)
    max_problems: PositiveInt | None = None
    inference: CodexInferenceConfig = Field(default_factory=CodexInferenceConfig)
    modal: ModalAppsConfig = Field(default_factory=ModalAppsConfig)
    generation_workers: PositiveInt = 8
    grading_workers: PositiveInt = 16

    @model_validator(mode="after")
    def validate_cipher_capacity(self) -> Self:
        """Require representable payload lengths and the existing prompt's example alphabet."""
        if self.sampling.max_bits.bit_length() > self.cipher.length_bits:
            raise ValueError("cipher.length_bits cannot represent sampling.max_bits")
        if not any(len(names) == 2 for names in self.cipher.special_variables.values()):
            raise ValueError("The prompt renderer requires a two-name synonym group")
        return self


class RunManifest(Record):
    """manifest.json: immutable metadata published when preparation finishes.

    schema_version: Literal 1, the artifact contract version.
    run_id: Safe directory component identifying this run.
    created_at: Timezone-aware preparation timestamp; producers write UTC.
    repository_commit: Source commit hash; None if unavailable.
    config: Fully resolved RunConfig, including dataset revision and cipher.
    problem_count/request_count: Prepared record counts, not completed-stage counts.

    The producer publishes this file last, after problems.jsonl and requests.jsonl,
    so its presence marks a prepared run. Consumers must reject missing manifests.
    The schema checks the count formula; the producer must check actual file counts.
    """

    schema_version: Literal[1] = 1
    run_id: Identifier
    created_at: AwareDatetime
    repository_commit: str | None
    config: RunConfig
    problem_count: NonnegativeInt
    request_count: NonnegativeInt

    @model_validator(mode="after")
    def validate_request_count(self) -> Self:
        """Require the recorded count to match the per-problem sampling budget."""
        sampling = self.config.sampling
        per_problem = sampling.lengths_per_problem * sampling.bitstrings_per_length + sampling.ordinary_generations_per_problem
        if self.request_count != self.problem_count * per_problem:
            raise ValueError("request_count does not match the sampling budget")
        return self


class ProblemRecord(Record):
    """One problems.jsonl line; problem_id joins to request records within a run.

    problem_id: APPS integer identifier within the manifest's split/revision.
    difficulty: Original APPS difficulty tier.
    public_problem: Existing AppsPromptProblem with question, starter_code, fn_name;
        these are the only problem fields the prompt builder may receive.
    test_cases: Existing AppsTestCases with paired inputs/outputs and fn_name.
        Persisted for Modal grading, never passed to Codex. Reference answers are
        not stored because this pipeline does not use them after eligibility filtering.
    """

    problem_id: NonnegativeInt
    difficulty: Literal["introductory", "interview", "competition"]
    public_problem: AppsPromptProblem
    test_cases: AppsTestCases

    @model_validator(mode="after")
    def validate_invocation(self) -> Self:
        """Require nonempty grading cases and matching public/evaluator entry points."""
        if not self.test_cases.inputs:
            raise ValueError("Trajectory grading requires at least one test case")
        if self.public_problem.fn_name != self.test_cases.fn_name:
            raise ValueError("Public prompt and evaluator must use the same fn_name")
        return self


class RequestRecord(Record):
    """One requests.jsonl line, persisted before any candidate inference.

    request_id: Unique safe candidate-directory component, stable on resume.
    problem_id: Join to ProblemRecord in this run.
    condition: ordinary means no steganography instructions, framed means explicit
        cipher/control instructions. This is a requested treatment, not a detector
        label or proof that a model intentionally encoded a message.
    ordinary_sample_index: Zero-based ordinary replicate index; None for framed.
    length_sample_index/bitstring_sample_index: Zero-based draw indices within the
        problem and length draw, respectively; they distinguish duplicate draws.
    sampled_length/message_bits: Drawn length and literal payload, including leading
        zeroes. Stored even for control_bit=0, but omitted from that prompt.
    control_bit: 0 requests explicit absence; 1 requests the exact payload.
    prompt: Exact nonblank text to submit once to infer(response_format="python").
        Ordinary requests have None for every framed sampling/target field; their
        prompt is build_apps_prompt(public_problem) with no secret argument.

    The schema enforces payload length. The producer enforces ID uniqueness, index
    bounds from RunConfig, and prompt/cipher/target agreement across records.
    """

    request_id: Identifier
    problem_id: NonnegativeInt
    condition: Condition
    ordinary_sample_index: NonnegativeInt | None
    length_sample_index: NonnegativeInt | None
    bitstring_sample_index: NonnegativeInt | None
    sampled_length: Annotated[int, Field(ge=1, le=8, strict=True)] | None
    message_bits: Payload | None
    control_bit: ControlBit | None
    prompt: str = Field(min_length=1, pattern=r"\S", strict=True)

    @model_validator(mode="after")
    def validate_payload_length(self) -> Self:
        """Separate ordinary/framed targets and preserve exact binary payload length."""
        framed_values = (self.length_sample_index, self.bitstring_sample_index, self.sampled_length, self.message_bits, self.control_bit)
        if self.condition == "ordinary":
            if self.ordinary_sample_index is None or any(value is not None for value in framed_values):
                raise ValueError("Ordinary requests require an ordinary index and null framed fields")
        elif self.ordinary_sample_index is not None or any(value is None for value in framed_values):
            raise ValueError("Framed requests require all framed fields and no ordinary index")
        elif len(self.message_bits) != self.sampled_length:
            raise ValueError("sampled_length must equal len(message_bits)")
        return self


class GenerationRecord(Record):
    """candidates/<request_id>/generation.json: one completed inference attempt.

    request_id: Join to RequestRecord; generation.json absence means unfinished.
    status: completed includes malformed answers; error means infrastructure failure.
    text/code: Exact raw response and extracted source. code=None means extraction
        failed; invalid Python source is retained when extraction succeeded.
    output_error: Format/syntax failure for a completed answer, otherwise None.
    error: Infrastructure exception description for status=error, otherwise None.
    resolved_model/turn_id: Backend identifiers when known; None before resolution.
    sdk_artifact_subdir: Optional pointer below STEGO_ARTIFACTS_DIR to the existing
        helper's request.json/answer.json/workspace; its native absolute paths are
        not copied into this public record. Native answer includes SDK restrictions,
        preflight details and observed event types for diagnostics.
    started_at/finished_at: Aware timestamps; producers write UTC.
    elapsed_seconds: Nonnegative wall duration, including preflight.

    The producer writes code.py as exact UTF-8 code when code is not None, with no
    reformatting or newline insertion. Schema validation does not parse or run code.
    Error retries replace this record only on explicit retry; completed malformed
    answers are never resampled. Native SDK attempt directories remain available.
    """

    request_id: Identifier
    status: Literal["completed", "error"]
    text: str | None
    code: str | None
    output_error: str | None
    error: str | None
    resolved_model: str | None
    turn_id: str | None
    sdk_artifact_subdir: Path | None
    started_at: AwareDatetime
    finished_at: AwareDatetime
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("sdk_artifact_subdir")
    @classmethod
    def validate_artifact_path(cls, value: Path | None) -> Path | None:
        """Reject absolute or parent-traversing artifact pointers; return a relative path."""
        if value is not None and (value.is_absolute() or ".." in value.parts):
            raise ValueError("SDK path must be relative to STEGO_ARTIFACTS_DIR")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Separate malformed responses from infrastructure failures and check timing."""
        if self.finished_at < self.started_at:
            raise ValueError("finished_at precedes started_at")
        if self.status == "error":
            if not self.error or self.output_error is not None:
                raise ValueError("Infrastructure errors require error and no output_error")
        elif self.error is not None or self.text is None:
            raise ValueError("Completed generation requires text and no infrastructure error")
        elif self.output_error is None and (self.code is None or not self.code.strip()):
            raise ValueError("A completed valid answer requires nonblank code")
        return self


class GradeRecord(Record):
    """candidates/<request_id>/grade.json: separate functional and framing outcomes.

    request_id: Join to RequestRecord and GenerationRecord.
    condition: Copy of RequestRecord.condition, so consumers can interpret nullable
        success fields. ordinary is a prompt treatment, not a ground-truth detector
        judgment about the resulting code.
    status: completed means candidate verdicts; error means grading infrastructure
        failure. Missing grade.json means ungraded, including failed generation.
    code_sha256: Hex SHA-256 of exact UTF-8 code, or None if no source was extracted.
    functional_success: Every supplied APPS case passed; false for malformed output.
    encoding_success: Exact present payload for control 1, explicit absent frame for
        control 0; false for malformed output or decoder failure. Missing control
        is a decoder failure, not a valid absent frame. None for ordinary requests,
        which have no requested encoding target, even when a frame decodes by chance.
    joint_success: AND of functional_success and encoding_success for framed
        requests; None for ordinary requests.
    verdict: Existing ModalAppsResult; None when malformed output skips execution
        or infrastructure fails before a verdict. Preserve raw verdicts/logs.
    decoded: Existing DecodedMessage, including bindings; None on skipped/failed
        decoding. Its payload is compared with framed RequestRecord by the producer.
        Ordinary decoding is diagnostic only and cannot establish encoding intent.
    decode_error: Decoder exception description, or None if successful/skipped.
    error: Infrastructure exception description, present only for status=error.
    elapsed_seconds: Nonnegative total grading duration.

    All three success fields are None for infrastructure errors so downstream
    summaries cannot count unfinished grading as an incorrect candidate. This model
    checks joint/null/error consistency; grade_candidate checks request-dependent
    success, hash correctness, and the correspondence with source and Modal verdict.
    """

    request_id: Identifier
    condition: Condition
    status: Literal["completed", "error"]
    code_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    functional_success: bool | None = Field(strict=True)
    encoding_success: bool | None = Field(strict=True)
    joint_success: bool | None = Field(strict=True)
    verdict: ModalAppsResult | None
    decoded: DecodedMessage | None
    decode_error: str | None
    error: str | None
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_success(self) -> Self:
        """Reject ambiguous infrastructure outcomes and inconsistent joint verdicts."""
        outcomes = (self.functional_success, self.encoding_success, self.joint_success)
        if self.status == "error":
            if not self.error or any(value is not None for value in outcomes):
                raise ValueError("Grading errors require error and null success fields")
        else:
            if self.error is not None or self.functional_success is None:
                raise ValueError("Completed grading requires a functional outcome and no error")
            if self.condition == "ordinary" and (self.encoding_success is not None or self.joint_success is not None):
                raise ValueError("Ordinary grading has no encoding or joint success target")
            if self.condition == "framed" and (self.encoding_success is None or self.joint_success is None):
                raise ValueError("Framed grading requires boolean encoding and joint outcomes")
            if self.condition == "framed" and self.joint_success != (self.functional_success and self.encoding_success):
                raise ValueError("joint_success must equal functional_success AND encoding_success")
            if self.verdict is not None and self.verdict.status == "runner_error":
                raise ValueError("Modal runner_error must be recorded as a grading error")
        if self.decoded is not None and self.decode_error is not None:
            raise ValueError("Successful decoding cannot also have a decode_error")
        return self


class StageSummary(Record):
    """Return value of a completed stage invocation; not a new artifact format.

    stage: Which stage ran. eligible_count: Number of input units (problems for
    prepare, requests otherwise). completed_count: Units completed this invocation,
    including incorrect/malformed candidates. skipped_count: Already-completed or
    ineligible units (e.g. generation errors cannot be graded). error_count: Units
    attempted this invocation that failed through infrastructure. elapsed_seconds:
    Total wall duration. Counts partition eligible_count; interruptions raise rather
    than returning a summary that pretends pending work completed.
    """

    stage: Stage
    eligible_count: NonnegativeInt
    completed_count: NonnegativeInt
    skipped_count: NonnegativeInt
    error_count: NonnegativeInt
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Require a completed invocation to account for every eligible unit."""
        if self.completed_count + self.skipped_count + self.error_count != self.eligible_count:
            raise ValueError("Stage counts must partition eligible_count")
        return self
