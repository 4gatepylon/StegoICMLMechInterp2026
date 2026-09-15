"""Small Codex inference, APPS prompt, and pass@k helpers.

Candidate execution belongs to modal_apps; combining its verdict with static
secret decoding currently belongs to the demonstration notebook.
"""

import ast
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
from openai_codex.async_client import AsyncCodexClient
from openai_codex.generated.v2_all import ConfigReadResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator, validate_call

from ciphers.variable_naming_in_python_v2.data.apps import REPO_ROOT
from ciphers.variable_naming_in_python_v2.decoder import CipherConfig


class CodexInferenceConfig(BaseModel):
    """Model, end-to-end SDK deadline, and storage for a fresh inference thread.

    None model uses the configured Codex default. Artifacts live beneath
    STEGO_ARTIFACTS_DIR/artifact_subdir/<uuid>; each request has an empty workspace.
    Subscription authentication, read-only sandboxing, denied approvals, and the
    tool restrictions below are fixed for this demonstration, not tunable here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str | None = None
    timeout_s: int = Field(default=180, ge=1, le=1800, strict=True)
    artifact_subdir: Path = Path("datasets/apps/codex-generation")

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        """Keep artifacts under their root and reject unusable model identifiers."""
        if self.artifact_subdir.is_absolute() or ".." in self.artifact_subdir.parts:
            raise ValueError("artifact_subdir must be relative to STEGO_ARTIFACTS_DIR without '..'")
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be a nonblank model ID or None")
        return self


class AppsPromptProblem(BaseModel):
    """Only public problem fields accepted by the prompt builder.

    question includes the public specification/examples; starter_code is optional
    scaffolding. fn_name is the evaluator's entry point for call-based questions,
    or None for stdin/stdout scripts. The caller obtains it from AppsTestCases.
    Hidden tests and reference answers are deliberately not fields of this model.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1)
    starter_code: str = ""
    fn_name: str | None = None


class SecretTask(BaseModel):
    """The requested payload and the existing decoder's complete alphabet.

    message_bits preserves leading zeroes and may be empty. This experiment fixes
    framing to one control bit, two length bits, and zero to three payload bits.
    CipherConfig itself supports other length widths; this demo rejects them.
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
    """The structured response schema: code is the complete Python source.

    Syntax and blank-source checks happen after generation in infer so malformed
    answers are saved and counted as failed samples rather than disappearing.
    """

    model_config = ConfigDict(extra="forbid")
    code: str


class InferenceResult(BaseModel):
    """Persisted result of one completed SDK turn, valid or malformed.

    text is the raw final response, empty if absent. code is the exact extracted
    source for Python responses, including syntactically invalid source when
    extractable; it is None for text responses or malformed JSON. output_error
    describes unusable output and must count as a failed candidate, not a retry.
    prompt/turn_id/requested_model identify the request. artifact_dir is absolute
    beneath STEGO_ARTIFACTS_DIR and contains request.json and answer.json.
    config_overrides and sandbox/approval_mode record requested restrictions;
    item_types records SDK turn events, not an inventory of available tools.
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


def _frame_example(secret: SecretTask, payload: str) -> str:
    """Illustrate framing with independent lambda-parameter bindings.

    secret supplies the alphabet; payload is a valid example bit string. Return
    source text, never executed. Separate lambda scopes allow repeated symbols
    to emit again even with a single synonym group. Any last symbol is padded
    with zeroes; the decoder ignores those bits after the framed payload.
    """
    names = next(iter(secret.cipher.special_variables.values()))
    width = (len(names) - 1).bit_length()
    frame = f"1{len(payload):02b}{payload}"
    padded = frame.ljust(math.ceil(len(frame) / width) * width, "0")
    return "\n".join(f"(lambda {names[int(padded[i : i + width], 2)]}: {names[int(padded[i : i + width], 2)]})(0)" for i in range(0, len(padded), width))


def build_apps_prompt(problem: AppsPromptProblem, *, secret: SecretTask | None = None) -> str:
    """Render a native-interface Python prompt, optionally with a secret task.

    problem contains only the three public fields documented by AppsPromptProblem;
    callers must not place references or private tests inside those fields.
    secret supplies the exact cipher and target payload; None requests ordinary
    code. Return the complete prompt for infer(response_format='python'). Jinja
    templates live beside this module; alphabet mappings and two static examples
    are generated from the supplied cipher, including multi-bit synonym groups.
    """
    templates = Environment(
        loader=FileSystemLoader(REPO_ROOT / "ciphers/variable_naming_in_python_v2/data/prompts"),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    alphabet = []
    examples = []
    if secret is not None:
        for group, names in secret.cipher.special_variables.items():
            width = (len(names) - 1).bit_length()
            alphabet.append((group, [(name, f"{index:0{width}b}") for index, name in enumerate(names)]))
        examples = [(payload, _frame_example(secret, payload)) for payload in ("", secret.message_bits or "01")]
    template = "apps_python.jinja2" if secret is None else "apps_secret.jinja2"
    return templates.get_template(template).render(problem=problem, secret=secret, alphabet=alphabet, examples=examples)


async def _disabled_mcp_overrides(runtime: CodexConfig) -> tuple[str, ...]:
    """Read effective configuration without creating a model thread.

    runtime uses the same cwd/environment as inference. Return only TOML overrides
    disabling each inherited MCP server; never persist or log the rest of the
    configuration, which may contain credentials. A separate short-lived public
    low-level SDK client is needed because AsyncCodex has no config-read method.
    No configuration is written back to disk.
    """
    async with AsyncCodexClient(runtime) as client:
        await client.initialize()
        response = await client.request("config/read", {"cwd": runtime.cwd, "includeLayers": False}, response_model=ConfigReadResponse)
    servers = (response.config.model_extra or {}).get("mcp_servers", {})
    # The CLI splits dotted override keys literally; quote server names inside
    # a TOML table VALUE so punctuation does not create a different server.
    disabled_servers = ", ".join(json.dumps(name) + "={enabled=false}" for name in sorted(servers))
    return ("mcp_servers={" + disabled_servers + "}",)


async def infer(prompt: str, config: CodexInferenceConfig, *, response_format: Literal["text", "python"] = "text") -> InferenceResult:
    """Run one prompt in a fresh ephemeral Codex thread using the existing login.

    prompt is sent unchanged; config controls the model, deadline, and artifact
    location. text format is for basic questions; python requests JSON with a code
    field. Return an InferenceResult and save it before any subsequent grading.
    Generated source is only AST-parsed here; execution belongs to Modal.

    Requires STEGO_ARTIFACTS_DIR and a ChatGPT-backed Codex login (subscription
    access depends on the account). API-key accounts are rejected and billing
    key environment variables are omitted. No application-level retry occurs.
    Authentication/transport/deadline/failed-turn errors propagate and should stop
    the experiment. Completed malformed answers instead return output_error so
    callers retain them in n. Unexpected non-message/reasoning turn items raise
    after saving the answer: this detects a restriction violation, not a new
    permission boundary. Settings target the pinned openai-codex SDK version.
    """
    if not prompt.strip() or response_format not in ("text", "python"):
        raise ValueError("Provide a nonblank prompt and text or python response_format")
    artifact_root = os.environ.get("STEGO_ARTIFACTS_DIR")
    if not artifact_root:
        raise ValueError("Set STEGO_ARTIFACTS_DIR before inference")
    artifact_dir = (REPO_ROOT / artifact_root / config.artifact_subdir / uuid4().hex).resolve()
    workspace = artifact_dir / "workspace"
    workspace.mkdir(parents=True)
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("CODEX_API_KEY", None)
    runtime = CodexConfig(cwd=str(workspace), env=environment, config_overrides=TOOL_RESTRICTIONS)
    (artifact_dir / "request.json").write_text(json.dumps({"prompt": prompt, "response_format": response_format, "config": config.model_dump(mode="json")}, indent=2))
    async with asyncio.timeout(config.timeout_s):
        overrides = TOOL_RESTRICTIONS + await _disabled_mcp_overrides(runtime)
        runtime = CodexConfig(cwd=str(workspace), env=environment, config_overrides=overrides)
        async with AsyncCodex(runtime) as codex:
            account = (await codex.account()).account
            if account is None or account.root.type != "chatgpt":
                raise RuntimeError("Sign in to Codex with ChatGPT; API-key authentication is not used")
            thread = await codex.thread_start(
                cwd=str(workspace),
                model=config.model,
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
        config_overrides=overrides,
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
    """Estimate oracle success among k candidates using Codex paper equation 1.

    n is the number of independently sampled, unfiltered candidates for ONE
    problem and fixed prompt; c counts successes under the chosen predicate;
    k is the subset size (1 <= k <= n). Return 1 - C(n-c,k)/C(n,k), evaluated
    with the paper's NumPy product implementation. Reject c > n or k > n. Malformed
    completed outputs belong in n as failures; infrastructure errors should stop
    the experiment instead of being classified as model failures. For multiple
    problems, compute this separately and average; do not pool their counts.

    This measures whether a subset contains a success, not whether a deployable
    ranking rule selects it. In particular k=n is simply float(c > 0).
    Source: https://arxiv.org/abs/2107.03374 (equation 1).
    """
    if c > n or k > n:
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
