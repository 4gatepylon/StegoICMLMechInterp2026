"""Generate standalone APPS stdin/stdout solutions with the Codex Python SDK.

The SDK reuses an existing ChatGPT-backed Codex login. Supplied reference answers
and hidden cases are never included in the generation prompt. Python source is
returned as structured output and parsed for syntax locally, but executed only
by the separate Modal evaluator.
"""

import ast
import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Self
from uuid import uuid4

from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
from pydantic import BaseModel, ConfigDict, Field, Json, model_validator

from ciphers.variable_naming_in_python_v2.data.apps import REPO_ROOT, AppsTestCases


class CodexAppsConfig(BaseModel):
    """Configure one subscription-authenticated generation, with no local execution.

    ``model=None`` lets Codex select its configured default; a model ID overrides
    that choice. ``timeout_s`` bounds SDK startup/authentication and generation.
    ``artifact_subdir`` is relative to STEGO_ARTIFACTS_DIR. Each request creates a
    separate empty working directory there, without copying the repository or
    APPS cache. Local execution tools, plugins, hooks, apps, and web search are
    disabled for this SDK process; the prompt additionally requests no tool use.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str | None = None
    timeout_s: int = Field(default=180, ge=1, le=1800, strict=True)
    artifact_subdir: Path = Path("datasets/apps/codex-generation")

    @model_validator(mode="after")
    def validate_artifact_path(self) -> Self:
        """Return this configuration after ensuring its path stays relative to the artifact root."""
        if self.artifact_subdir.is_absolute() or ".." in self.artifact_subdir.parts:
            raise ValueError("artifact_subdir must be relative to STEGO_ARTIFACTS_DIR without '..'")
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be a nonblank model ID or None")
        return self


class _PromptProblem(BaseModel):
    """Fields consumed from a load_apps result when building a generation prompt.

    Required ``question`` is the natural-language specification; ``starter_code``
    is source-provided scaffolding, possibly empty; ``input_output`` is either
    an AppsTestCases mapping or its JSON string representation and is inspected
    only to require fn_name=None (stdio mode).
    Other row fields, particularly ``solutions``, are ignored. Hidden input and
    output values are validated but not serialized into the prompt.
    """

    model_config = ConfigDict(extra="ignore", strict=True)
    question: str = Field(min_length=1)
    starter_code: str
    input_output: AppsTestCases | Json[AppsTestCases]


class PythonAnswer(BaseModel):
    """Structured Codex output: ``code`` is the entire standalone Python program.

    Extra keys, blank source, and invalid Python syntax are rejected. The schema
    is sent through Thread.run(output_schema=...), then this model validates the
    final JSON response. Parsing syntax does not execute source or establish
    correctness; Modal subsequently evaluates its behavior on supplied cases.
    """

    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_python_source(self) -> Self:
        """Return a syntactically valid nonblank answer without importing or executing it."""
        if not self.code.strip():
            raise ValueError("Codex returned blank source")
        try:
            ast.parse(self.code)
        except SyntaxError as error:
            raise ValueError(f"Codex returned invalid Python: {error.msg}") from error
        return self


class CodexAppsAnswer(BaseModel):
    """Generation artifact consumed by the notebook and evaluate_on_modal.

    ``code`` is a syntax-checked program, passed unchanged to Modal. ``prompt``
    records exactly what was sent, excluding references/hidden tests. ``turn_id``
    identifies the SDK turn. ``requested_model`` is nullable because Codex may
    choose its default. ``artifact_dir`` is the run's absolute path beneath the
    configured artifact root; answer.json there stores this same record. The
    record contains no authentication credentials or account identifiers.
    """

    code: str
    prompt: str
    turn_id: str
    requested_model: str | None
    artifact_dir: str


def build_stdio_prompt(problem: Mapping[str, object]) -> str:
    """Create a generation prompt for a standard-input/output APPS loader row.

    Args:
        problem: Mapping with the required fields documented by _PromptProblem.
            Function-call problems are rejected: this demo deliberately asks for
            a script with the APPS task's native stdin/stdout protocol. Public
            examples already present in question are preserved.

    Returns:
        Instructions plus the full question and any starter scaffold. References,
        hidden cases, source URL, and problem ID are not added. The prompt requires
        exact output formatting, no diagnostic output, and no local tool use.
    """
    row = _PromptProblem.model_validate(problem)
    if row.input_output.fn_name is not None:
        raise ValueError("This generation demo requires a standard-input/output APPS problem")
    if not row.question.strip():
        raise ValueError("The problem description must not be blank")
    scaffold = f"\n\nStarter scaffold supplied by the problem:\n{row.starter_code}" if row.starter_code else ""
    return (
        "Implement the programming problem below as one complete Python 3.10 program.\n"
        "Read input from standard input using exactly the format described in the problem.\n"
        "Write only the required answer to standard output, respecting the requested line/token format.\n"
        "If the problem specifies multiple test cases, process all of them in one invocation.\n"
        "Do not read filenames, fetch data, hardcode sample answers, or print explanations/debug logs.\n"
        "Use the Python standard library. Do not use tools, inspect files, or run code; solve from this prompt.\n"
        "Return the complete program in the JSON code field required by the output schema, without Markdown fences.\n\n"
        f"Problem:\n{row.question}{scaffold}"
    )


async def generate_stdio_solution(problem: Mapping[str, object], config: CodexAppsConfig | None = None) -> CodexAppsAnswer:
    """Generate one APPS program through the pip-installed official Codex SDK.

    Args:
        problem: A loader row accepted by build_stdio_prompt. Ground-truth code
            and hidden cases are never submitted to the model.
        config: Model, deadline, and artifact settings; None uses defaults.

    Returns:
        A CodexAppsAnswer, also saved as answer.json under its artifact directory
        before the caller attempts Modal execution. response.json additionally
        stores status (SDK turn-state string) and final_response (nullable raw
        answer text), so malformed structured output can be inspected afterwards.
        Authentication must report
        a ChatGPT account; API-key accounts and logged-out sessions raise without
        starting a generation. Subscription capacity/access is determined by
        Codex, not asserted by this helper. SDK, timeout, and output-validation
        errors propagate; there is no API-key fallback or automatic retry.

    Usage:
        Await this function from Jupyter, then pass answer.code and the loader
        row's AppsTestCases to evaluate_on_modal. STEGO_ARTIFACTS_DIR must be set;
        a relative value resolves against the repository root. The SDK uses its
        packaged CLI runtime and the user's existing login without copying tokens.
    """
    config = config if config is not None else CodexAppsConfig()
    prompt = build_stdio_prompt(problem)
    artifact_root = os.environ.get("STEGO_ARTIFACTS_DIR")
    if not artifact_root:
        raise ValueError("Set STEGO_ARTIFACTS_DIR before generating an APPS solution")
    artifact_dir = (REPO_ROOT / artifact_root / config.artifact_subdir / uuid4().hex).resolve()
    workspace = artifact_dir / "workspace"
    workspace.mkdir(parents=True)
    # Keep the SDK's existing subscription login; do not pass API billing keys.
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("CODEX_API_KEY", None)
    runtime = CodexConfig(
        cwd=str(workspace),
        env=environment,
        config_overrides=(
            'model_provider="openai"',
            'web_search="disabled"',
            "features.shell_tool=false",
            "features.unified_exec=false",
            "features.plugins=false",
            "features.hooks=false",
            "features.apps=false",
            "features.multi_agent=false",
            "features.code_mode=false",
            "features.code_mode_host=false",
        ),
    )
    async with asyncio.timeout(config.timeout_s):
        async with AsyncCodex(runtime) as codex:
            account = (await codex.account()).account
            if account is None or account.root.type != "chatgpt":
                raise RuntimeError("Sign in to Codex with ChatGPT before running this subscription demo; API-key authentication is not used")
            thread = await codex.thread_start(
                cwd=str(workspace),
                model=config.model,
                model_provider="openai",
                approval_mode=ApprovalMode.deny_all,
                sandbox=Sandbox.read_only,
                ephemeral=True,
                base_instructions="Solve the supplied coding problem from the prompt alone. Do not use any tools or execute code. Return the requested structured answer.",
            )
            turn = await thread.run(prompt, output_schema=PythonAnswer.model_json_schema())
    (artifact_dir / "response.json").write_text(json.dumps({"status": turn.status.value, "final_response": turn.final_response}, indent=2))
    if turn.status.value != "completed" or not turn.final_response:
        raise RuntimeError(f"Codex did not complete an answer: {turn.status}")
    answer = PythonAnswer.model_validate_json(turn.final_response)
    result = CodexAppsAnswer(
        code=answer.code,
        prompt=prompt,
        turn_id=turn.id,
        requested_model=config.model,
        artifact_dir=str(artifact_dir),
    )
    (artifact_dir / "answer.json").write_text(result.model_dump_json(indent=2))
    return result
