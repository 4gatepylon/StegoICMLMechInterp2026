"""Evaluate APPS source remotely with a pinned upstream evaluator on Modal.

The local process only prepares JSON and reads verdicts. Each solution gets a
fresh CPU Sandbox with network access blocked and no user secrets mounted.
The upstream runner, including its permissive output comparisons, is unchanged.
"""

import io
import json
import tokenize
import urllib.request
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from ciphers.variable_naming_in_python_v2.data.apps import AppsTestCases

EVALUATOR_REVISION = "b45c0ed78517a3a6492eb77b21cffbb79b1096f1"
EVALUATOR_URL = f"https://raw.githubusercontent.com/hendrycks/apps/{EVALUATOR_REVISION}/eval/testing_util.py"
# The remote artifact root is independent of the local notebook's artifact root.
REMOTE_ARTIFACTS_DIR = "/stego-artifacts"
REMOTE_DRIVER_PATH = Path(__file__).resolve().parent / "sandbox_remote_drivers" / "modal_remote_driver.py"
REMOTE_DRIVER_FILENAME = "modal_remote_driver.py"
EVALUATOR_PATH = REMOTE_DRIVER_PATH.with_name("apps_evaluator.py")


def _python_code_tokens(source: str) -> list[tuple[int, str]]:
    """Return Python token types/text for comparing source without comments.

    ``source`` is Python source text, never executed here. The returned ordered
    pairs omit comments, non-significant newlines, and end markers; token offsets
    are excluded so inserted comment lines do not affect equality. Significant
    newlines and indentation remain, as do every string literal and docstring
    (including any ``#`` characters). Inter-token spacing is ignored by Python's
    tokenizer. Tokenization/indentation errors propagate; this is source-token
    equivalence, not a claim that differently written programs behave alike.
    """
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER}
    return [(token.type, token.string) for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type not in ignored]


def verified_evaluator_source() -> str:
    """Read the annotated evaluator and verify it against pinned GitHub source.

    Takes no arguments: EVALUATOR_PATH and EVALUATOR_URL identify the local file
    and immutable upstream revision. Each call reads both afresh, with a 30-second
    HTTP timeout and no application cache. The exact returned local text is what
    evaluate_on_modal uploads, avoiding a second file read after verification.

    Raises AssertionError on a token mismatch, including changes to docstrings,
    literals, or indentation. The explicit raise remains active under Python -O.
    File, HTTP, decoding, and tokenizer errors propagate before any Modal resource
    is created. Comments/blank lines may differ; neither source is executed here.
    """
    local_source = EVALUATOR_PATH.read_text(encoding="utf-8")
    with urllib.request.urlopen(EVALUATOR_URL, timeout=30) as response:
        upstream_source = response.read().decode("utf-8")
    if _python_code_tokens(local_source) != _python_code_tokens(upstream_source):
        raise AssertionError(f"Local APPS evaluator differs from pinned upstream code: {EVALUATOR_URL}")
    return local_source


class ModalAppsConfig(BaseModel):
    """Bound one remote solution evaluation and select its Modal application.

    ``case_timeout_s`` sets the upstream evaluator's integer alarm for compilation
    and each case. Its -1 verdict combines runtime errors and case timeouts.
    ``solution_timeout_s`` limits the entire evaluator process; a separate Sandbox
    lifetime adds 30 seconds for setup. Neither limit includes image build time.
    ``memory_mb`` is the Sandbox memory limit; only one CPU is requested.
    ``max_log_chars`` truncates returned diagnostics, not program output before
    comparison. ``app_name`` identifies a lazily created Modal app; credentials
    come from the caller's normal Modal environment/configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_name: str = Field(default="stego-apps-evaluation", min_length=1)
    case_timeout_s: int = Field(default=4, ge=1, le=60, strict=True)
    solution_timeout_s: int = Field(default=120, ge=1, le=3600, strict=True)
    memory_mb: int = Field(default=1024, ge=256, strict=True)
    max_log_chars: int = Field(default=4000, ge=1, le=100_000, strict=True)


class _WorkerOutput(BaseModel):
    """Validate the JSON emitted by the remote driver on its stdout channel.

    Required ``results`` contains upstream verdicts: True/False per completed
    case, -1 for a runtime error or case timeout, or a singleton -2 for an
    initialization/compile failure. ``logs`` is a truncated diagnostic string;
    ``error`` is None or an exception escaping the upstream evaluator. Early
    initialization failure can return fewer verdicts than supplied cases.
    """

    model_config = ConfigDict(extra="forbid")
    results: list[StrictBool | Literal[-1, -2]]
    logs: str
    error: str | None


class ModalAppsResult(BaseModel):
    """One solution's verdict, suitable for display and JSON artifact storage.

    ``status`` is passed only when every supplied case explicitly passed, failed
    for upstream negative verdicts, timeout for Modal's process deadline, or
    runner_error for missing/invalid results or evaluator infrastructure failure.
    ``num_tests`` counts supplied cases; ``passed_tests`` counts literal True
    verdicts, never the truthiness of the negative upstream error codes.
    ``raw_results`` preserves upstream codes (see ``_WorkerOutput``).
    ``logs`` and nullable ``error`` explain failures. ``sandbox_id`` identifies
    the already-terminated Sandbox for diagnostics; it is not a live resource.
    """

    model_config = ConfigDict(extra="forbid")
    status: Literal["passed", "failed", "timeout", "runner_error"]
    num_tests: int
    passed_tests: int = 0
    raw_results: list[StrictBool | Literal[-1, -2]] = Field(default_factory=list)
    logs: str = ""
    error: str | None = None
    sandbox_id: str


def interpret_verdict(stdout: str, num_tests: int, sandbox_id: str) -> ModalAppsResult:
    """Validate a remote stdout JSON message and classify its upstream verdicts.

    Args:
        stdout: JSON with exactly the ``_WorkerOutput`` schema. Invalid JSON or
            schema errors propagate, allowing the caller to mark a runner error.
        num_tests: Positive number of cases submitted; missing verdicts can never
            produce a pass, including an empty list or early compilation failure.
        sandbox_id: Modal resource ID to include in the returned diagnostic record.

    Returns:
        A ``ModalAppsResult`` with counts and original verdict codes. Upstream
        exceptions take precedence over partial passes. An unexplained verdict
        count mismatch is a runner error; singleton -2 is a documented compile
        failure and is classified as failed.
    """
    if num_tests <= 0:
        raise ValueError("At least one supplied test case is required")
    output = _WorkerOutput.model_validate_json(stdout)
    passed = sum(value is True for value in output.results)
    error = output.error
    if not error and len(output.results) != num_tests and output.results != [-2]:
        error = f"Expected {num_tests} verdicts, received {len(output.results)}"
    status = "runner_error" if error else "passed" if passed == num_tests else "failed"
    return ModalAppsResult(
        status=status,
        num_tests=num_tests,
        passed_tests=passed,
        raw_results=output.results,
        logs=output.logs,
        error=error,
        sandbox_id=sandbox_id,
    )


def evaluator_image() -> modal.Image:
    """Describe the remote Python 3.10 image; building occurs only when Modal runs.

    Returns:
        A Modal image with numpy==1.26.4 and pyext==0.6 installed and import-checked.
        Python 3.10 retains inspect.getargspec, required by pyext. The artifact
        directory is created at build time; verified evaluator source and the
        worker are uploaded separately on every invocation. No local credentials
        or repository files are copied into this image.
    """
    return (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install("numpy==1.26.4", "pyext==0.6")
        .env({"STEGO_ARTIFACTS_DIR": REMOTE_ARTIFACTS_DIR})
        .run_commands(f"mkdir -p {REMOTE_ARTIFACTS_DIR}")
        .workdir(REMOTE_ARTIFACTS_DIR)
        .run_commands("python -c 'import numpy, pyext'")
    )


def evaluate_on_modal(code: str, test_cases: AppsTestCases, config: ModalAppsConfig | None = None) -> ModalAppsResult:
    """Run one Python solution against all supplied APPS cases in a fresh Sandbox.

    Args:
        code: Nonblank Python source, either a supplied reference or a generated
            answer. It is serialized as data locally and executed only on Modal.
        test_cases: Validated ``AppsTestCases`` from a loader row's ``input_output``.
            fn_name=None selects standard input/output; otherwise upstream invokes
            that function or Solution method. Empty test lists are rejected.
        config: Remote resource/time limits, or ``None`` for ``ModalAppsConfig()``.

    Returns:
        A ``ModalAppsResult`` describing upstream comparisons and diagnostics.
        The Sandbox is always terminated and detached after creation, including
        upload, execution, and result-parsing failures. Authentication, image-build,
        and transport errors propagate; they are not mislabeled as wrong answers.
        Every call verifies the local evaluator against pinned GitHub source;
        source mismatches and verification/network failures raise before Modal
        lookup. The verified source and worker are uploaded next to request.json;
        neither is baked into the image. The upstream comparator is
        permissive (including numeric and unordered fallbacks); this is an
        APPS-compatibility demo, not a hardened judge. TODO(hadriano) what would it mean for this to be
        a "hardened judge"?
    """
    if not code.strip() or not test_cases.inputs:
        raise ValueError("A nonblank solution and at least one supplied case are required")
    config = config if config is not None else ModalAppsConfig()
    evaluator_source = verified_evaluator_source()
    app = modal.App.lookup(config.app_name, create_if_missing=True)
    sandbox = modal.Sandbox.create(
        app=app,
        image=evaluator_image(),
        timeout=config.solution_timeout_s + 30,
        cpu=1,
        memory=(config.memory_mb, config.memory_mb),
        block_network=True,
    )
    try:
        request = {
            "code": code,
            "input_output": test_cases.model_dump(),
            "case_timeout_s": config.case_timeout_s,
            "max_log_chars": config.max_log_chars,
        }
        sandbox.filesystem.write_text(json.dumps(request), f"{REMOTE_ARTIFACTS_DIR}/request.json")
        sandbox.filesystem.write_text(evaluator_source, f"{REMOTE_ARTIFACTS_DIR}/apps_evaluator.py")
        sandbox.filesystem.write_text(REMOTE_DRIVER_PATH.read_text(), f"{REMOTE_ARTIFACTS_DIR}/{REMOTE_DRIVER_FILENAME}")
        process = sandbox.exec("python", f"{REMOTE_ARTIFACTS_DIR}/{REMOTE_DRIVER_FILENAME}", timeout=config.solution_timeout_s)
        exit_code = process.wait()
        if exit_code == -1:
            # Modal 1.5 returns -1 from ContainerProcess.wait on ExecTimeoutError.
            return ModalAppsResult(status="timeout", num_tests=len(test_cases.inputs), sandbox_id=sandbox.object_id, error="Modal evaluator process deadline exceeded")
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        if exit_code:
            return ModalAppsResult(
                status="runner_error",
                num_tests=len(test_cases.inputs),
                sandbox_id=sandbox.object_id,
                logs=(stdout + stderr)[-config.max_log_chars :],
                error=f"Evaluator process exited with code {exit_code}",
            )
        try:
            return interpret_verdict(stdout, len(test_cases.inputs), sandbox.object_id)
        except ValueError as error:
            return ModalAppsResult(
                status="runner_error",
                num_tests=len(test_cases.inputs),
                sandbox_id=sandbox.object_id,
                logs=(stdout + stderr)[-config.max_log_chars :],
                error=f"Invalid evaluator result: {error}",
            )
    finally:
        try:
            sandbox.terminate()
        finally:
            sandbox.detach()
