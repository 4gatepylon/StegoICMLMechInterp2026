"""Public API contracts for the future trajectory pipeline; all bodies are stubs.

Import artifact/configuration models from trajectory_generation.schemas. This
module performs no I/O at import time. No stub calls Codex, downloads APPS, or
creates Modal resources. Every function raises NotImplementedError until the
implementation PR replaces its body. Docstrings describe required future behavior.
"""

from collections.abc import Iterator
from pathlib import Path
from random import Random

from ciphers.variable_naming_in_python_v2.data.codex_apps import AppsPromptProblem
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig
from ciphers.variable_naming_in_python_v2.trajectory_generation.schemas import (
    GenerationRecord,
    GradeRecord,
    ProblemRecord,
    RequestRecord,
    RunConfig,
    RunManifest,
    SamplingConfig,
    Stage,
    StageSummary,
)


def sample_length(rng: Random, *, min_bits: int, max_bits: int) -> int:
    """Draw one uniform length from the inclusive bounds, with replacement.

    rng is the caller-owned local generator, never the global random state.
    min_bits/max_bits must satisfy 1 <= min_bits <= max_bits <= 8. Return an int
    used as the length of every payload in this length draw. Reject invalid bounds
    with ValueError. Keep length sampling separate so a future distribution change
    does not require changing prompt construction, grading, or artifact consumers.
    """
    raise NotImplementedError("Trajectory sampling is not implemented")


def sample_bits(rng: Random, *, length: int, one_probability: float) -> str:
    """Draw an IID Bernoulli payload, preserving leading zeroes in the result.

    rng is the caller-owned generator; length is an integer in 1..8;
    one_probability is finite and in [0, 1]. Return exactly length characters from
    '0'/'1'. Reject invalid inputs with ValueError. In particular 0 and 1 produce
    all-zero and all-one strings. Do not hardcode a fair coin in this function.
    """
    raise NotImplementedError("Trajectory sampling is not implemented")


def sample_control_bit(rng: Random, *, one_probability: float) -> int:
    """Draw the independent presence bit using an explicit Bernoulli parameter.

    rng is the caller-owned generator and one_probability must be finite in [0, 1].
    Return integer 0 or 1, consumed as RequestRecord.control_bit. Invalid inputs
    raise ValueError. This probability is independent of payload_one_probability;
    callers may force absent/present controls with endpoint values 0 and 1.
    """
    raise NotImplementedError("Trajectory sampling is not implemented")


def estimate_codex_calls(problem_count: int, sampling: SamplingConfig) -> int:
    """Return the planned number of model turns before retries or resume skips.

    problem_count is the nonnegative number of selected APPS rows after filtering
    and max_problems. sampling supplies the length-draw and bitstring-draw counts.
    Return problem_count * (lengths_per_problem * bitstrings_per_length +
    ordinary_generations_per_problem). Both conditions use the same model. It is exact
    for prepared requests; it is an estimate of actual remote calls because failures
    may stop submission. Metadata/preflight calls are not model turns. A generation
    resume separately counts pending records and explicitly requested error retries.
    The CLI must print both total prepared and scheduled-this-invocation counts
    before any generation, and print zero for grading-only invocations.
    """
    raise NotImplementedError("Trajectory call estimation is not implemented")


def build_request_prompt(problem: AppsPromptProblem, cipher: CipherConfig, *, control_bit: int, message_bits: str) -> str:
    """Render one exact prompt while withholding hidden tests and reference answers.

    problem contains only public question, starter_code, and fn_name. cipher is
    shared with the decoder. control_bit is integer 0/1; message_bits is a nonempty
    1..8-bit string representable by cipher.length_bits, including leading zeroes.
    Return the prompt stored in RequestRecord and later submitted without changes.

    For control 1, import build_apps_prompt and SecretTask from data.codex_apps;
    they already import the builders in data.prompts. For control 0, reuse the
    ordinary APPS prompt and extend the shared data prompt builders with an explicit
    absent-frame requirement. Do not include the sampled payload or length, or
    present-message-only instructions, in an absent prompt. A zero control needs
    no length/payload fields; no emitted bindings is an error, not valid absence.
    Do not copy the existing worked examples into a second template implementation.
    """
    raise NotImplementedError("Trajectory prompt adaptation is not implemented")


def sample_requests(problem: ProblemRecord, config: RunConfig, *, rng: Random) -> Iterator[RequestRecord]:
    """Yield every request for one problem, including duplicate sampled targets.

    problem supplies public prompt fields and the problem_id join key. config
    supplies sampling and cipher settings. rng is shared across problems visited
    in ascending problem_id order and seeded once from config.sampling.seed.
    For each length draw, draw a length; for each bitstring draw, draw all payload
    bits then an independent control bit. Use the three public sampling functions.

    First yield ordinary_generations_per_problem ordinary RequestRecord values,
    with IDs 'p{problem_id}-o{ordinary_sample_index}', all framed fields None, and
    build_apps_prompt(problem.public_problem) unchanged. Ordinary requests do not
    consume random draws. Then yield lengths_per_problem * bitstrings_per_length
    framed RequestRecord values with zero-based draw indices and exact prompts
    from build_request_prompt. Framed IDs are
    deterministic 'p{problem_id}-l{length_sample_index}-b{bitstring_sample_index}',
    unique within a run even when targets repeat. Do not execute or grade anything.
    Exact bit sequences across Python/runtime versions are not promised; persisted
    request records are authoritative for replay. Invalid inputs raise ValueError.
    """
    raise NotImplementedError("Trajectory request preparation is not implemented")


def prepare_run(run_id: str, config: RunConfig) -> RunManifest:
    """Filter APPS, persist all requests, and publish the immutable run manifest.

    run_id must satisfy schemas.Identifier and names a directory below
    STEGO_ARTIFACTS_DIR/datasets/apps/trajectory-generation. config is the resolved
    master configuration. Use load_apps(config.apps), sort by problem_id, then apply
    max_problems. Reuse AppsTestCases.from_dataset_value to parse supplied tests.
    Resolve config.inference.artifact_subdir to this run's sdk/ subdirectory in the
    returned manifest; no other stored settings change.

    Return RunManifest after atomically publishing problems.jsonl, requests.jsonl,
    then manifest.json last. Print the expected model-call count and a tqdm bar
    for request preparation. Refuse to overwrite any existing run directory;
    interruption requires a new run_id or deliberate cleanup of the partial run.
    No credentials/model calls/Modal submissions are needed. Empty selection is a
    valid zero-request run. Environment/storage/download errors propagate.
    """
    raise NotImplementedError("Trajectory preparation is not implemented")


def load_manifest(run_id: str) -> RunManifest:
    """Read the prepared run's immutable metadata for generation or review tools.

    run_id is a safe Identifier under the fixed artifact layout, not a filesystem
    path. Resolve STEGO_ARTIFACTS_DIR relative to the repository root when necessary.
    Return RunManifest parsed from manifest.json; missing/unprepared runs raise
    FileNotFoundError, schema violations raise pydantic.ValidationError. Do not
    create directories, download data, or infer missing metadata. Manifest version
    1 is required; later incompatible layouts must use a new schema version.
    """
    raise NotImplementedError("Trajectory artifact reading is not implemented")


async def generate_candidate(request: RequestRecord, config: RunConfig) -> GenerationRecord:
    """Adapt the existing single-request inference helper into a stable disk record.

    request supplies the exact persisted prompt and join ID. config is the manifest's
    resolved master configuration. Await data.codex_apps.infer with response_format
    'python'; retain its restrictions, preflight policy, and native SDK artifacts.
    Return a GenerationRecord containing exact text/code and elapsed wall time.
    Translate the helper's absolute artifact_dir into an artifact-root-relative
    sdk_artifact_subdir. Return status=error for service exceptions; retain completed
    malformed answers as status=completed with output_error. Never resample them.

    This function performs generation but does not write the public candidate files;
    generate_run persists returned records atomically. Cancellation propagates.
    Source is never executed locally. Existing inference may save native evidence
    before raising; preserve its pointer when available, otherwise use None.
    """
    raise NotImplementedError("Trajectory generation is not implemented")


async def generate_run(run_id: str, *, retry_errors: bool = False) -> StageSummary:
    """Generate pending requests with bounded concurrency and durable per-item output.

    run_id selects a prepared manifest/requests file. retry_errors=False skips any
    existing generation record; True also schedules status=error records. Always
    skip status=completed records, even malformed answers. The manifest config
    controls model and generation_workers; no sampling or config changes on resume.

    Before submitting, print prepared/scheduled/skipped model-turn counts. Use a
    bounded worker queue and completion-driven tqdm bar with throughput and ETA.
    Write code.py before generation.json, using temporary files plus atomic replace.
    Missing generation.json means incomplete even if code.py already exists.
    On quota/auth/model prerequisite failure, stop scheduling, drain in-flight work,
    persist available results, and raise; do not report a complete StageSummary.
    Return a StageSummary otherwise, retaining infrastructure errors separately
    from candidate failures. Cancellation/storage/schema errors propagate. A run
    must have only one writer at a time; simultaneous CLI writers are unsupported.
    """
    raise NotImplementedError("Trajectory generation is not implemented")


async def grade_candidate(request: RequestRecord, problem: ProblemRecord, generation: GenerationRecord, config: RunConfig) -> GradeRecord:
    """Grade one completed answer with the existing decoder and Modal evaluator.

    request supplies condition and optional expected control/payload; problem supplies test_cases;
    generation supplies unchanged source and output validity; config supplies cipher
    and Modal settings. Require matching request/problem IDs and completed generation,
    raising ValueError on mismatches. SHA-256 covers exact source encoded as UTF-8.

    Return GradeRecord: malformed answers fail functional_success and, for framed
    requests, encoding_success/joint_success; skip Modal/decoding and retain the
    generation.output_error in generation.json.
    Otherwise call decode locally and evaluate_on_modal remotely without changing
    code. control=1 requires is_encoding plus exact length and payload; control=0
    requires is_encoding=False. DecodeError fails encoding independently of tests.
    All tests must pass for functional_success; framed joint_success is their
    conjunction. Ordinary answers are decoded for diagnostics, but encoding_success
    and joint_success stay None: there was no requested message. An accidentally
    decodable frame does not establish intentional steganography.
    Service exceptions and runner_error produce status=error with null success
    fields. Preserve any available verdict/decoded result. Cancellation propagates.

    Extract the notebook's evaluate_candidate logic here, extending it for explicit
    absent frames and durable infrastructure outcomes. This function returns the
    record; grade_run owns persistence. Candidate code executes only on Modal.
    """
    raise NotImplementedError("Trajectory grading is not implemented")


async def grade_run(run_id: str, *, retry_errors: bool = False) -> StageSummary:
    """Grade generated candidates concurrently using persisted inputs only.

    run_id selects the manifest, problems, requests, and generation records.
    retry_errors=False skips existing grade records; True resubmits status=error
    grades only. Skip absent/failed generation records. Grade completed malformed
    answers as failures without Modal execution. Use config.grading_workers for a
    bounded queue, tqdm completion counts/ETA, and atomic per-candidate grade.json
    writes. Return StageSummary; storage/schema errors and cancellation propagate.
    Announce zero Codex calls. Existing candidate failures are never retried to seek
    a passing answer. All model output is read from disk without regeneration.
    """
    raise NotImplementedError("Trajectory grading is not implemented")


async def run_pipeline(run_id: str, *, steps: tuple[Stage, ...], config: RunConfig | None = None, retry_errors: bool = False) -> tuple[StageSummary, ...]:
    """Run requested stages in prepare/generate/grade order through one entry point.

    run_id selects the fixed artifact directory. steps must be a nonempty, unique
    selection of stage names; order is normalized to dependency order. config is
    required exactly when prepare is selected, otherwise load the saved manifest.
    retry_errors applies only to generate/grade. Preparation contributes a summary
    with selected problems as completed units. Return one StageSummary per requested
    stage in execution order. Stages are sequential; workers parallelize candidates
    within a stage. Missing prerequisites and invalid options raise rather than
    implicitly preparing or generating. All bodies remain stubs in this PR.
    """
    raise NotImplementedError("Trajectory orchestration is not implemented")


def load_config(config_path: Path) -> RunConfig:
    """Parse the single master JSON configuration without ad hoc dictionary merging.

    config_path is relative to the repository root. Return RunConfig via Pydantic
    JSON validation; omitted fields use its documented defaults. Reject absolute
    paths and parent traversal. File errors and pydantic.ValidationError propagate.
    The CLI's explicitly supplied overrides must be represented as typed Pydantic
    settings and revalidated before prepare; omitted CLI options never replace
    values from this file. Resume uses the manifest rather than rereading this file.
    """
    raise NotImplementedError("Trajectory config loading is not implemented")
