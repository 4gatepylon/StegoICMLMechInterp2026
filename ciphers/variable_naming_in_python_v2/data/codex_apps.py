"""Subscription-backed Codex inference, APPS prompts, preflight checks, and pass@k.

Public interface:
    CodexInferenceConfig: Model (default Luna), deadline, storage, and quota policy.
    preflight(config): Check prerequisites; return PreflightResult or raise
        CodexPreflightError with a failed stage and corrective hint.
    infer(prompt, config, response_format="text" | "python"): Async generation;
        automatically preflight, save artifacts, and return InferenceResult.
    build_apps_prompt(problem, secret=None): Render AppsPromptProblem plus optional
        SecretTask as Markdown; SecretTask uses the existing decoder's CipherConfig.
    pass_at_k(n, c, k): Codex paper's oracle-success estimator.

Example (after setting STEGO_ARTIFACTS_DIR and signing in to Codex with ChatGPT):
    from ciphers.variable_naming_in_python_v2.data.codex_apps import CodexInferenceConfig, infer

    config = CodexInferenceConfig()
    answer = await infer("What is the capital of France?", config)
    print(answer.text)

Supports text/Python responses, native call-based or stdin/stdout APPS prompts,
and 0–3-bit secret prompts. Does not provide API-key billing, configurable thinking
modes, candidate execution, decoding, or batch orchestration. The notebook combines
Modal execution and the existing decoder; the helper generates without tool use.
"""

import ast
import asyncio
import json
import os
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from tempfile import TemporaryFile
from typing import Annotated, Literal, Self, override
from uuid import uuid4

import numpy as np
from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, CodexError, Sandbox
from openai_codex.async_client import AsyncCodexClient
from openai_codex.generated.v2_all import ConfigReadResponse, GetAccountRateLimitsResponse, ModelListResponse, RateLimitSnapshot
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator, validate_call

from ciphers.variable_naming_in_python_v2.data.apps import REPO_ROOT
from ciphers.variable_naming_in_python_v2.data.prompts import build_python_prompt, build_secret_prompt
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig


class CodexInferenceConfig(BaseModel):
    """Configure model selection, request deadlines, artifacts, and usage checks.

    Pass this immutable Pydantic model to preflight or infer. Unknown fields and
    invalid field values are rejected during construction; network-dependent
    checks such as model availability and remaining quota run in preflight.

    Attributes:
        model: Exact Codex model ID; defaults to "gpt-5.6-luna". A nonblank string
            selects that model. None resolves the user's effective SDK model setting,
            falling back to the catalog's default only when that setting is absent.
            Preflight validates the resolved ID, which infer then submits explicitly.
        timeout_s: Integer SDK deadline in seconds, from 1 through 1800; defaults
            to 180. For infer, it covers preflight, SDK startup/authentication, and
            the model turn together. Standalone preflight uses the same duration
            for its SDK metadata calls. It does not set Modal execution limits or
            bound result-file persistence after the turn finishes.
        artifact_subdir: Directory relative to STEGO_ARTIFACTS_DIR; defaults to
            "datasets/apps/codex-generation" to group these experiment records.
            Absolute paths and '..' components are rejected. Each inference creates
            a UUID directory beneath it for request.json, answer.json, and an
            initially empty workspace/. Preflight creates/checks the base directory
            without creating a candidate request. STEGO_ARTIFACTS_DIR must be set;
            a relative environment value resolves against REPO_ROOT.
        min_remaining_usage_percent: Required percentage remaining, from 0 through
            100, in every reported window of usage_limit_id. Defaults to 10.0 to
            retain some subscription headroom; equality passes. Zero removes the
            reserve requirement but still checks backend blocks and quota data.
            None skips usage inspection entirely, while retaining storage, login,
            and model checks. Enabled inspection fails when the selected quota
            data is unavailable. This is a snapshot, not a budget reservation or
            an estimate of how much the upcoming request will consume.
        usage_limit_id: Exact SDK quota-bucket ID to inspect; defaults to "codex",
            the general Codex bucket. It is independent of model and is not inferred
            from the model name. Set it explicitly for a different reported bucket.
            All reported primary/secondary windows in that bucket are checked;
            their durations need not be daily or weekly. This field has no effect
            when min_remaining_usage_percent is None.

    Notes:
        ChatGPT subscription authentication, read-only permissions, denied approvals,
        and the tool restrictions are fixed by the helper rather than configured
        here. Reasoning effort is not yet exposed by this configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str | None = "gpt-5.6-luna"
    # TODO(hadriano): Add reasoning_effort, validate model support in preflight, and pass it to thread.run(effort=...).
    timeout_s: int = Field(default=180, ge=1, le=1800, strict=True)
    artifact_subdir: Path = Path("datasets/apps/codex-generation")
    min_remaining_usage_percent: float | None = Field(default=10.0, ge=0, le=100, allow_inf_nan=False)
    usage_limit_id: str = Field(default="codex", min_length=1)

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        """Keep artifacts under their root and reject unusable model identifiers."""
        if self.artifact_subdir.is_absolute() or ".." in self.artifact_subdir.parts:
            raise ValueError("artifact_subdir must be relative to STEGO_ARTIFACTS_DIR without '..'")
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be a nonblank model ID or None")
        return self


class AppsPromptProblem(BaseModel):
    """Public APPS fields accepted by the prompt builder.

    Attributes:
        question: Nonempty public problem specification, including public examples.
        starter_code: Public Python scaffold; an empty string means none was supplied.
        fn_name: Call-based evaluator entry point from AppsTestCases. None means the
            candidate must use stdin/stdout instead of returning a function result.

    Notes:
        Hidden tests and reference answers are excluded from this schema. Callers
        must also keep them out of the public strings above.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1)
    starter_code: str = ""
    fn_name: str | None = None


class SecretTask(BaseModel):
    """Pair a requested payload with the existing decoder's cipher.

    Attributes:
        cipher: Validated CipherConfig containing the ordered synonym groups and
            framing settings. This demo additionally requires length_bits=2;
            CipherConfig already fixes control_bits=1. Prompt rendering also requires
            a two-name group to construct an example with exactly the frame's bits.
        message_bits: Literal binary payload of length 0–3. Leading zeroes matter;
            an empty string requests a present empty message, not an absent frame.
        frame_bits: Computed control + length + payload string. For example, payload
            "101" produces "111101", while an empty payload produces "100".
    """

    model_config = ConfigDict(extra="forbid")
    cipher: CipherConfig
    message_bits: str = Field(pattern=r"^[01]*$", max_length=3, strict=True)

    @model_validator(mode="after")
    def validate_frame(self) -> Self:
        """Require the framing requested by this experiment; return this task."""
        if self.cipher.length_bits != 2:
            raise ValueError("This demo requires exactly two length bits")
        return self

    @property
    def frame_bits(self) -> str:
        """Return present-control + big-endian length + exact payload bits."""
        return f"1{len(self.message_bits):02b}{self.message_bits}"


class PythonAnswer(BaseModel):
    """Schema supplied to the SDK for a structured Python response.

    Attributes:
        code: Complete generated Python source. This schema rejects extra fields;
            infer separately checks blank source and syntax after generation so
            malformed answers can be saved and counted as failed samples.
    """

    model_config = ConfigDict(extra="forbid")
    code: str


class CodexPreflightError(RuntimeError):
    """Report a failed prerequisite with a machine-readable stage and remedy.

    Attributes:
        stage: Failed check: artifacts, sdk, authentication, model, or usage.
        hint: Corrective action, also included in the formatted exception message.
        __cause__: Original OS/provider exception when one is chained by preflight;
            None for failures detected directly, such as an unavailable model ID.
    """

    @override
    def __init__(self, stage: str, detail: str, hint: str) -> None:
        """Attach a failed check and remedy in addition to RuntimeError's message.

        Args:
            stage: Identifier of the failed prerequisite check.
            detail: Concrete failure description used in str(error).
            hint: Suggested corrective action, stored separately for higher-level callers.
        """
        self.stage = stage
        self.hint = hint
        super().__init__(f"Codex preflight [{stage}]: {detail} Fix: {hint}")


class UsageWindow(BaseModel):
    """Record one backend-reported quota window.

    Attributes:
        name: Window role, either primary or secondary, within the selected bucket.
        remaining_percent: Percentage remaining, calculated as 100 minus the SDK's
            used percentage. Validated to lie between 0 and 100 inclusive.
        window_duration_mins: Window duration in minutes; None means it was omitted.
            Primary/secondary roles do not imply daily/weekly durations.
        resets_at: Reset time in Unix seconds; None means it was omitted.
    """

    name: Literal["primary", "secondary"]
    remaining_percent: float = Field(ge=0, le=100)
    window_duration_mins: int | None
    resets_at: int | None


class PreflightResult(BaseModel):
    """Record a credential-free snapshot before generation starts.

    Attributes:
        model: Resolved catalog model that infer will submit explicitly.
        artifact_base_dir: Writable absolute directory beneath STEGO_ARTIFACTS_DIR;
            each request creates its UUID directory inside it.
        config_overrides: Non-secret SDK overrides, including tool restrictions and
            disabled inherited MCP servers.
        usage_limit_id: Selected quota-bucket ID; not inspected when usage is skipped.
        usage_windows: Checked primary/secondary windows. Empty only when the
            configured reserve is None, explicitly skipping usage inspection.
        checked_at: Time the checks completed, as a timezone-aware UTC datetime.

    Notes:
        Passing does not reserve quota or guarantee a later inference will succeed.
    """

    model: str
    artifact_base_dir: Path
    config_overrides: tuple[str, ...]
    usage_limit_id: str
    usage_windows: tuple[UsageWindow, ...]
    checked_at: datetime


class InferenceResult(BaseModel):
    """Persist one completed SDK turn, including malformed responses.

    Attributes:
        text: Raw final response; an empty string means no final text was supplied.
        code: Extracted Python source, preserving invalid syntax when extractable.
            None means a text response or JSON from which no code could be extracted.
        output_error: Output-validation failure description, or None on success.
            A completed Python answer with an error counts as a failed sample.
        prompt: Exact prompt sent to the model.
        turn_id: SDK identifier for the completed turn.
        requested_model: Caller-requested model ID; None means the caller requested
            the configured default. The resolved ID is recorded in preflight.model.
        artifact_dir: Absolute request directory beneath STEGO_ARTIFACTS_DIR,
            containing request.json, answer.json, and workspace/.
        config_overrides: Non-secret restrictions requested for the SDK process.
        sandbox: Requested filesystem policy; fixed to "read-only" by this helper.
        approval_mode: Requested approval policy; fixed to "deny_all" by this helper.
        item_types: Observed SDK turn-event types, not an inventory of available tools.
        preflight: Resolved model, storage location, restrictions, and quota snapshot
            checked before this request started.
    """

    text: str
    code: str | None
    output_error: str | None
    prompt: str
    turn_id: str
    requested_model: str | None
    artifact_dir: str
    config_overrides: tuple[str, ...]
    sandbox: Literal["read-only"] = "read-only"
    approval_mode: Literal["deny_all"] = "deny_all"
    item_types: tuple[str, ...]
    preflight: PreflightResult


# Read-only permissions do not disable tools. These overrides separately remove
# execution/integration features, and per-server overrides below disable MCPs.
TOOL_RESTRICTIONS = (
    'model_provider="openai"',
    'web_search="disabled"',
    "tools.view_image=false",
    "features.shell_tool=false",
    "features.unified_exec=false",
    "features.plugins=false",
    "features.hooks=false",
    "features.apps=false",
    "features.multi_agent=false",
    "features.code_mode=false",
    "features.code_mode_host=false",
)


def build_apps_prompt(problem: AppsPromptProblem, *, secret: SecretTask | None = None) -> str:
    """Build a Markdown Python prompt, optionally adding a secret-message task.

    Args:
        problem: Validated public APPS fields. The caller must not put reference
            answers or hidden tests inside those fields.
        secret: Validated cipher and payload; None requests ordinary code generation.

    Returns:
        str: Complete prompt for infer(response_format="python"), assembled by the
            Python builders in data/prompts.py with the task's native interface.

    Raises:
        ValueError: A secret cipher has no two-name group for the exact-bit example.
            Add such a group to cipher.special_variables before rendering the prompt.
    """
    prompt = build_python_prompt(problem.question, problem.starter_code, problem.fn_name)
    if secret is not None:
        prompt += "\n\n" + build_secret_prompt(secret.cipher, secret.message_bits)
    return prompt


def _runtime(workspace: Path, overrides: tuple[str, ...]) -> CodexConfig:
    """Build SDK settings without modifying the parent's environment.

    Args:
        workspace: Artifact-backed working directory for the SDK process.
        overrides: Non-secret CLI settings, including the requested tool restrictions.

    Returns:
        CodexConfig: Child settings with the inherited environment and cleared API
            billing keys. Preflight and infer verify authentication separately.
    """
    environment = os.environ.copy()
    # Clear API billing keys for subscription auth; the SDK merges overrides into the parent environment.
    environment["OPENAI_API_KEY"] = ""
    environment["CODEX_API_KEY"] = ""
    return CodexConfig(cwd=str(workspace), env=environment, config_overrides=overrides)


def _check_usage(response: GetAccountRateLimitsResponse, config: CodexInferenceConfig) -> tuple[UsageWindow, ...]:
    """Enforce the enabled reserve policy against SDK quota metadata.

    Args:
        response: Typed account/rateLimits/read output. Multi-bucket values must
            follow RateLimitSnapshot, including its primary/secondary windows.
        config: Selects the exact quota bucket and a non-None reserve threshold.

    Returns:
        tuple[UsageWindow, ...]: All reported windows in the selected bucket, after
            each passes the reserve check; callers record them in PreflightResult.

    Raises:
        CodexPreflightError: An explicit backend block, unavailable bucket/windows,
            or a window below the requested reserve prevents generation.
        ValidationError: SDK metadata violates the window schema; preflight wraps
            this as a usage-stage CodexPreflightError and preserves the cause.

    Notes:
        Reset times, credit balances, and missing flags do not imply restored access.
    """
    if response.ordinary_usage_allowed is False:
        raise CodexPreflightError("usage", "The backend disallows ordinary included usage.", "Check Codex usage limits and wait for access to resume.")
    buckets = response.rate_limits_by_limit_id or {}
    if config.usage_limit_id in buckets:
        bucket = RateLimitSnapshot.model_validate(buckets[config.usage_limit_id])
    elif response.rate_limits.limit_id == config.usage_limit_id or (response.rate_limits.limit_id is None and config.usage_limit_id == "codex"):
        bucket = response.rate_limits
    else:
        raise CodexPreflightError("usage", f"No quota bucket {config.usage_limit_id!r}; available: {sorted(buckets)}.", "Set usage_limit_id to a reported bucket.")
    if bucket.spend_control_reached or bucket.rate_limit_reached_type is not None:
        reason = bucket.rate_limit_reached_type.value if bucket.rate_limit_reached_type else "spend control reached"
        raise CodexPreflightError("usage", f"Bucket {config.usage_limit_id!r} is blocked: {reason}.", "Check the account/workspace usage settings or wait for the limit to reset.")
    windows = []
    for name in ("primary", "secondary"):
        reported = getattr(bucket, name)
        if reported is None:
            continue
        window = UsageWindow(name=name, remaining_percent=100 - reported.used_percent, window_duration_mins=reported.window_duration_mins, resets_at=reported.resets_at)
        if window.remaining_percent < config.min_remaining_usage_percent:
            reset = datetime.fromtimestamp(window.resets_at, timezone.utc).isoformat() if window.resets_at is not None else "unknown"
            raise CodexPreflightError(
                "usage",
                f"{config.usage_limit_id}/{name} ({window.window_duration_mins or 'unknown'} minutes) has "
                f"{window.remaining_percent:g}% remaining; requires {config.min_remaining_usage_percent:g}%. Reset: {reset}.",
                "Wait for the reset or explicitly lower min_remaining_usage_percent.",
            )
        windows.append(window)
    if not windows:
        raise CodexPreflightError(
            "usage",
            f"No usage windows reported for {config.usage_limit_id!r}; the reserve cannot be verified.",
            "Retry later, or set min_remaining_usage_percent=None to explicitly skip usage inspection.",
        )
    return tuple(windows)


async def preflight(config: CodexInferenceConfig) -> PreflightResult:
    """Check prerequisites without starting a model turn or consuming inference quota.

    Args:
        config: Model, SDK deadline, artifact directory, and reserve policy. None
            model resolves the effective SDK default; None reserve skips quota RPCs.

    Returns:
        PreflightResult with the resolved model, writable artifact base, safe tool
        overrides, checked usage windows, and check time. infer repeats this check
        for every candidate and records it with the answer; callers may also call
        it separately to diagnose setup before starting a larger experiment.

    Preconditions:
        STEGO_ARTIFACTS_DIR must name a writable location (relative values resolve
        against REPO_ROOT). The SDK must be installed, connected, and logged in
        through ChatGPT. The requested model must occur in the SDK model catalog.

    Postconditions:
        The artifact base directory exists and a temporary write probe has been
        removed. No generation thread, candidate directory, or persistent SDK
        configuration is created. The metadata client is closed. Passing establishes
        a snapshot of setup/usage, not a quota reservation or a promise of inference
        access. Thresholds apply to every reported window of the selected bucket;
        missing duration/reset metadata is preserved, not guessed.

    Raises:
        CodexPreflightError: A setup, SDK, authentication, model, or usage check
            fails. Its stage and hint support callers several layers up; underlying
            OS/SDK causes are chained. Errors include a concrete corrective action.
    """
    stage = "artifacts"
    hints = {
        "artifacts": "Set STEGO_ARTIFACTS_DIR to a writable directory and check filesystem permissions/free space.",
        "sdk": "Check the installed openai-codex version, network connection, and Codex configuration.",
        "authentication": "Sign in to Codex with ChatGPT and retry; API-key accounts are not supported here.",
        "model": "Choose an exact model ID from the Codex model catalog, such as gpt-5.6-luna.",
        "usage": "Check Codex usage availability; retry later or explicitly disable the reserve with min_remaining_usage_percent=None.",
    }
    try:
        artifact_root = os.environ.get("STEGO_ARTIFACTS_DIR")
        if not artifact_root:
            raise CodexPreflightError(stage, "STEGO_ARTIFACTS_DIR is not set.", hints[stage])
        base = (REPO_ROOT / artifact_root / config.artifact_subdir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        with TemporaryFile(dir=base) as probe:
            probe.write(b"Codex artifact write check")
            probe.flush()
        stage = "sdk"
        async with asyncio.timeout(config.timeout_s):
            async with AsyncCodexClient(_runtime(base, TOOL_RESTRICTIONS)) as client:
                await client.initialize()
                stage = "authentication"
                account = (await client.account_read()).account
                if account is None or account.root.type != "chatgpt":
                    raise CodexPreflightError(stage, "No ChatGPT-backed Codex login is active.", hints[stage])
                stage = "sdk"
                effective = await client.request("config/read", {"cwd": str(base), "includeLayers": False}, response_model=ConfigReadResponse)
                stage = "model"
                models = []
                cursor = None
                while True:
                    page = await client.request("model/list", {"includeHidden": True, "cursor": cursor}, response_model=ModelListResponse)
                    models.extend(page.data)
                    cursor = page.next_cursor
                    if cursor is None:
                        break
                model = config.model or effective.config.model or next((item.model for item in models if item.is_default), None)
                available = sorted({item.model for item in models})
                if model not in available:
                    suggestions = get_close_matches(model or "", available, n=3) or available[:5]
                    raise CodexPreflightError(stage, f"Model {model!r} is not in the Codex catalog.", f"Set config.model to an available ID; suggestions: {suggestions}.")
                servers = (effective.config.model_extra or {}).get("mcp_servers", {})
                # Dotted CLI keys split literally; put quoted server names inside a TOML value.
                disabled_servers = ", ".join(json.dumps(name) + "={enabled=false}" for name in sorted(servers))
                overrides = TOOL_RESTRICTIONS + ("mcp_servers={" + disabled_servers + "}",)
                windows = ()
                if config.min_remaining_usage_percent is not None:
                    stage = "usage"
                    usage = await client.request("account/rateLimits/read", {"excludeResetCreditDetails": True}, response_model=GetAccountRateLimitsResponse)
                    windows = _check_usage(usage, config)
        return PreflightResult(
            model=model, artifact_base_dir=base, config_overrides=overrides, usage_limit_id=config.usage_limit_id, usage_windows=windows, checked_at=datetime.now(timezone.utc)
        )
    except CodexPreflightError:
        raise
    except (OSError, CodexError, ValidationError) as error:
        raise CodexPreflightError(stage, f"{type(error).__name__}: {error}", hints[stage]) from error


async def infer(prompt: str, config: CodexInferenceConfig, *, response_format: Literal["text", "python"] = "text") -> InferenceResult:
    """Run one prompt in a fresh Codex thread and persist the result for evaluation.

    Args:
        prompt: Nonblank instructions sent unchanged to the model. For APPS code,
            use build_apps_prompt to supply the public task and optional secret
            instructions without reference answers or hidden tests.
        config: CodexInferenceConfig controlling the model, SDK deadline in seconds,
            and artifact subdirectory. Its default model is gpt-5.6-luna; an explicit
            None uses the user's configured Codex default. The deadline covers
            configuration discovery, startup, authentication, and the model turn;
            subsequent Modal grading has its own limits. The default reserve policy
            requires 10% remaining in every reported window of the codex quota bucket;
            min_remaining_usage_percent=None explicitly skips usage inspection.
        response_format: "text" returns an ordinary answer, useful for basic
            questions. "python" requests JSON with a single code field, extracts
            that field, and checks nonblank Python syntax without executing it.

    Returns:
        InferenceResult containing raw text, extracted code (None for text responses
        or unextractable JSON), and output_error (None when output validation passes).
        It also records the prompt, turn ID, requested model, artifact directory,
        requested restrictions, observed SDK item types, and the preflight snapshot.
        Callers must count a completed answer with output_error as a failed sample
        rather than dropping
        or retrying it. Otherwise, pass code unchanged to the Modal evaluator;
        successful syntax validation does not imply functional correctness.

    Preconditions:
        STEGO_ARTIFACTS_DIR must be set to a writable artifact root. Relative values
        resolve against REPO_ROOT. It separates generated experiment data from
        repository source and gives every request a durable record for inspection,
        reproduction, and diagnosing failures before or during subsequent grading.
        Each request uses <root>/<config.artifact_subdir>/<uuid>/, with an initially
        empty workspace/ as the SDK working directory; no dataset or repo is copied.

        The pinned openai-codex SDK must be installed and have a ChatGPT-backed
        login with access to the selected model. This experiment uses subscription
        capacity: API billing keys are cleared in the child environment and API-key
        accounts are rejected. Authentication is reused, never created here.

    Postconditions:
        Preflight runs before any model turn. request.json records the prompt,
        response_format, config, and preflight snapshot before generation starts.
        A completed turn's answer.json is saved before returning, including malformed
        output, so later grading failures cannot erase the generated answer. Failures before a
        completed turn may leave only request.json; preflight failures create no
        request. Both files remain for the caller to inspect or remove; the SDK
        process is closed when its context exits.

        Each request uses a fresh ephemeral thread with read-only permissions,
        denied approvals, and the tool restrictions for the pinned SDK. Unexpected
        non-message/reasoning items raise after saving the answer; this activity
        check is not itself a permission boundary. Candidate source is never run
        by this helper. There are no application-level retries or Modal submissions.

    Raises:
        ValueError: The prompt or response format is invalid. Configuration field
            validation occurs when constructing config.
        CodexPreflightError: Setup, login, model, or usage checks failed; inspect
            stage and hint for the cause and corrective action.
        TimeoutError: The SDK phase exceeds config.timeout_s.
        RuntimeError: Authentication is not ChatGPT-backed, the turn does not
            complete, or unexpected SDK item types violate the experiment contract.
        OSError: Artifact creation or persistence fails. SDK authentication and
            transport exceptions also propagate unchanged. Such infrastructure
            failures should stop the experiment, not count as incorrect solutions.
    """
    if not prompt.strip() or response_format not in ("text", "python"):
        raise ValueError("Provide a nonblank prompt and text or python response_format")
    async with asyncio.timeout(config.timeout_s):
        checks = await preflight(config)
        artifact_dir = checks.artifact_base_dir / uuid4().hex
        workspace = artifact_dir / "workspace"
        workspace.mkdir(parents=True)
        runtime = _runtime(workspace, checks.config_overrides)
        request = {"prompt": prompt, "response_format": response_format, "config": config.model_dump(mode="json"), "preflight": checks.model_dump(mode="json")}
        (artifact_dir / "request.json").write_text(json.dumps(request, indent=2))
        async with AsyncCodex(runtime) as codex:
            account = (await codex.account()).account
            if account is None or account.root.type != "chatgpt":
                raise RuntimeError("Sign in to Codex with ChatGPT; API-key authentication is not used")
            thread = await codex.thread_start(
                cwd=str(workspace),
                model=checks.model,
                model_provider="openai",
                approval_mode=ApprovalMode.deny_all,
                sandbox=Sandbox.read_only,
                ephemeral=True,
                base_instructions="Answer from the supplied prompt alone. Do not use tools, inspect files, or execute code. Follow the requested response format.",
            )
            turn = await thread.run(prompt, output_schema=PythonAnswer.model_json_schema() if response_format == "python" else None)
    if turn.status.value != "completed":
        raise RuntimeError(f"Codex did not complete an answer: {turn.status}")
    source = None
    output_error = None
    raw_text = turn.final_response or ""
    if response_format == "python":
        try:
            source = PythonAnswer.model_validate_json(raw_text).code
            if not source.strip():
                raise ValueError("Blank Python source")
            ast.parse(source)
        except (ValidationError, SyntaxError, ValueError) as error:
            output_error = str(error)
    elif not raw_text.strip():
        output_error = "Blank text response"
    result = InferenceResult(
        text=raw_text,
        code=source,
        output_error=output_error,
        prompt=prompt,
        turn_id=turn.id,
        requested_model=config.model,
        artifact_dir=str(artifact_dir),
        config_overrides=checks.config_overrides,
        preflight=checks,
        item_types=tuple(item.root.type for item in turn.items),
    )
    (artifact_dir / "answer.json").write_text(result.model_dump_json(indent=2))
    unexpected = set(result.item_types) - {"userMessage", "agentMessage", "reasoning"}
    if unexpected:
        raise RuntimeError(f"Unexpected Codex turn items {sorted(unexpected)}; inspect {artifact_dir / 'answer.json'}")
    return result


@validate_call
def pass_at_k(
    n: Annotated[int, Field(ge=1, strict=True)],
    c: Annotated[int, Field(ge=0, strict=True)],
    k: Annotated[int, Field(ge=1, strict=True)],
) -> float:
    """Estimate oracle success using the Codex paper's NumPy implementation.

    Args:
        n: Total independently sampled, unfiltered candidates for one fixed problem
            and prompt. Must be a positive integer.
        c: Number satisfying the chosen success predicate; must satisfy 0 <= c <= n.
            Completed malformed outputs belong in n and contribute nothing to c.
        k: Subset size; must be an integer satisfying 1 <= k <= n.

    Returns:
        float: 1 - C(n-c,k)/C(n,k), the probability a uniformly chosen k-subset contains
            a success. At k=n this is simply float(c > 0).

    Raises:
        ValidationError: Counts have invalid types or violate their individual bounds.
        ValueError: c or k exceeds n.

    Notes:
        Infrastructure failures should stop the experiment rather than count as
        incorrect answers. For multiple problems, average their separate estimates;
        do not pool counts. This measures oracle success, not a ranking method.
        Source: https://arxiv.org/abs/2107.03374 (equation 1).
    """
    if c > n or k > n:
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
