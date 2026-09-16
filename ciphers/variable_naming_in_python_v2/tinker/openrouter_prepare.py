"""Prepare a small OpenRouter screening run; future training happens on Tinker."""

import json
import math
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ciphers.variable_naming_in_python_v2.data.apps import REPO_ROOT, AppsConfig, AppsTestCases, load_apps
from ciphers.variable_naming_in_python_v2.data.codex_apps import AppsPromptProblem, SecretTask, build_apps_prompt

MODEL_IDS = (
    "nvidia/nemotron-3.5-lightning",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-nano-30b-a3b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-35b-a3b",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-4b",
    "thinkingmachines/inkling-small",
)
API_ROOT = "https://openrouter.ai/api/v1"


class RunConfig(BaseModel):
    """Settings saved before any inference.

    ``secret`` is the shared cipher/payload used in both prompting and decoding.
    ``models`` contains exact OpenRouter IDs; unavailable IDs are explicitly skipped.
    ``num_problems`` caps the shared shuffled APPS subset at 100. Each model gets
    one answer per problem, without repair or repeated sampling. ``seed`` only
    controls dataset selection. ``apps`` controls the existing APPS loader.
    ``max_tokens`` limits requested output (usually reasoning plus visible text).
    ``estimated_output_tokens`` is a cost assumption, not a generation setting.
    Model defaults control temperature and reasoning. ``timeout_s`` bounds each
    HTTP call, not the full experiment or Modal grading.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    secret: SecretTask
    models: tuple[str, ...] = Field(default=MODEL_IDS, min_length=1)
    num_problems: int = Field(default=100, ge=1, le=100, strict=True)
    seed: int = 42
    apps: AppsConfig = Field(default_factory=AppsConfig)
    max_tokens: int = Field(default=16384, ge=1, strict=True)
    estimated_output_tokens: int = Field(default=4096, ge=1, strict=True)
    timeout_s: int = Field(default=300, ge=1, strict=True)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        """Reject duplicate requests and inconsistent output-cost assumptions."""
        if len(set(self.models)) != len(self.models):
            raise ValueError("Model IDs must be unique")
        if self.estimated_output_tokens > self.max_tokens:
            raise ValueError("estimated_output_tokens must not exceed max_tokens")
        return self


class Pricing(BaseModel):
    """OpenRouter catalog USD per input/output token and optional per-request fee."""

    prompt: float = Field(ge=0, allow_inf_nan=False)
    completion: float = Field(ge=0, allow_inf_nan=False)
    request: float = Field(default=0, ge=0, allow_inf_nan=False)


class CatalogModel(BaseModel):
    """Catalog fields consumed by preparation; other upstream metadata is ignored."""

    id: str
    pricing: Pricing


class Message(BaseModel):
    """One public user prompt; private test cases never enter this schema."""

    role: Literal["user"] = "user"
    content: str


class RequestBody(BaseModel):
    """Exact chat body: model ID, one user message, and output-token limit.

    No tools, schema enforcement, or reasoning overrides are sent. All models
    receive the same prompt requesting a JSON object with a string ``code`` field.
    """

    model_config = ConfigDict(extra="forbid")
    model: str
    messages: tuple[Message]
    max_tokens: int = Field(ge=1)


class PreparedRequest(BaseModel):
    """JSONL row: unique request_id, APPS problem_id, and exact HTTP body."""

    model_config = ConfigDict(extra="forbid")
    request_id: str
    problem_id: int
    body: RequestBody


class ModelEstimate(BaseModel):
    """Per-model cost assumptions; unavailable models have zero requests and cost.

    ``input_characters`` counts message text, excluding JSON serialization.
    ``input_tokens`` estimates ceil(chars/3) + 16 chat-overhead tokens per request.
    ``estimated_usd`` uses the configured output assumption; ``limit_scenario_usd``
    uses max_tokens for every answer. Neither is a guaranteed billing ceiling.
    ``pricing`` preserves the catalog's per-token USD rates at preparation time.
    """

    model: str
    available: bool
    requests: int = 0
    input_characters: int = 0
    input_tokens: int = 0
    pricing: Pricing | None = None
    estimated_usd: float = 0
    limit_scenario_usd: float = 0


def artifact_path(relative: Path) -> Path:
    """Resolve a run path below STEGO_ARTIFACTS_DIR; reject absolute/escaping paths.

    ``relative`` is the value returned by prepare_run. A relative environment
    variable is resolved against REPO_ROOT. The returned absolute path is for IO.
    """
    root = (REPO_ROOT / os.environ["STEGO_ARTIFACTS_DIR"]).resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Run directory must be relative to STEGO_ARTIFACTS_DIR")
    return path


def fetch_catalog(model_ids: tuple[str, ...]) -> dict[str, CatalogModel]:
    """GET public model/pricing metadata without credentials or inference.

    ``model_ids`` selects exact IDs; unrelated router aliases can have unknown
    pricing sentinels and are not parsed. Returns available requested IDs mapped
    to validated CatalogModel records. The response must contain ``data``, a list
    with id and pricing on each requested model.
    HTTP, JSON, and schema errors propagate rather than yielding a partial catalog.
    """
    with urllib.request.urlopen(f"{API_ROOT}/models", timeout=30) as response:
        data = json.load(response)
    models = [CatalogModel.model_validate(row) for row in data["data"] if row["id"] in model_ids]
    return {model.id: model for model in models}


def estimate_cost(requests: list[PreparedRequest], config: RunConfig, catalog: dict[str, CatalogModel]) -> list[ModelEstimate]:
    """Estimate each configured model's costs from the saved request bodies.

    ``requests`` must be the prepared rows, ``config`` their generation settings,
    and ``catalog`` the fetch_catalog snapshot. Returns one ModelEstimate per
    configured model in order, including unavailable IDs. Costs ignore caching,
    provider price variation, account fees, and Modal. Output lengths are unknown;
    the two scenarios explicitly separate that uncertainty from input counting.
    """
    estimates = []
    for model_id in config.models:
        model = catalog.get(model_id)
        rows = [row for row in requests if row.body.model == model_id]
        chars = [len(row.body.messages[0].content) for row in rows]
        tokens = sum(math.ceil(n / 3) + 16 for n in chars)
        estimate = ModelEstimate(model=model_id, available=model is not None, requests=len(rows), input_characters=sum(chars), input_tokens=tokens)
        if model is not None:
            estimate.pricing = model.pricing
            input_cost = tokens * model.pricing.prompt + len(rows) * model.pricing.request
            estimate.estimated_usd = input_cost + len(rows) * config.estimated_output_tokens * model.pricing.completion
            estimate.limit_scenario_usd = input_cost + len(rows) * config.max_tokens * model.pricing.completion
        estimates.append(estimate)
    return estimates


def prepare_run(config: RunConfig) -> Path:
    """Save all inputs and an estimate without making inference/Modal requests.

    ``config`` selects models, data, and one shared secret. Returns a run directory
    relative to STEGO_ARTIFACTS_DIR for run_prepared and summarize. Files:
    config.json (RunConfig); requests.jsonl (PreparedRequest rows); grading_cases.json
    (string problem IDs mapped to AppsTestCases); estimate.json (timestamp, per-model
    ModelEstimate records, total estimated_usd, and total limit_scenario_usd).
    All prompts are written before estimating. Reference solutions are not saved.
    Fewer eligible problems than requested raises instead of silently shrinking N.
    """
    catalog = fetch_catalog(config.models)
    dataset = load_apps(config.apps).shuffle(seed=config.seed)
    if len(dataset) < config.num_problems:
        raise ValueError(f"Requested {config.num_problems} problems but only {len(dataset)} qualify")
    requests = []
    grading_cases = {}
    for problem in dataset.select(range(config.num_problems)):
        cases = AppsTestCases.from_dataset_value(problem["input_output"])
        if not cases.inputs:
            raise ValueError("Each problem must have at least one grading case")
        problem_id = problem["problem_id"]
        if str(problem_id) in grading_cases:
            raise ValueError(f"Duplicate problem ID: {problem_id}")
        grading_cases[str(problem_id)] = cases.model_dump(mode="json")
        public = AppsPromptProblem(question=problem["question"], starter_code=problem["starter_code"], fn_name=cases.fn_name)
        prompt = build_apps_prompt(public, secret=config.secret)
        prompt += '\n\nReturn exactly one JSON object: {"code": "<complete Python source>"}.'
        for model_id in config.models:
            if model_id in catalog:
                requests.append(
                    PreparedRequest(
                        request_id=f"{problem_id}:{model_id}",
                        problem_id=problem_id,
                        body=RequestBody(model=model_id, messages=(Message(content=prompt),), max_tokens=config.max_tokens),
                    )
                )
    now = datetime.now(timezone.utc)
    relative = Path("variable_naming_v2/tinker") / f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    directory = artifact_path(relative)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(config.model_dump_json(indent=2))
    (directory / "grading_cases.json").write_text(json.dumps(grading_cases, indent=2))
    (directory / "requests.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in requests))
    saved = [PreparedRequest.model_validate_json(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    estimates = estimate_cost(saved, config, catalog)
    (directory / "estimate.json").write_text(
        json.dumps(
            {
                "created_at": now.isoformat(),
                "models": [e.model_dump(mode="json") for e in estimates],
                "estimated_usd": sum(e.estimated_usd for e in estimates),
                "limit_scenario_usd": sum(e.limit_scenario_usd for e in estimates),
            },
            indent=2,
        )
    )
    return relative
