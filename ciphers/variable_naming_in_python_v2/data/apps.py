"""Load APPS questions with size-filtered reference answers and supplied test cases.

Only data is downloaded and parsed: neither APPS's loading script nor any answer
or test case is executed. ``load_apps`` uses the official Hugging Face repository's
converted Parquet files; ``filter_apps`` applies the same contract to local rows.
"""

import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from datasets import Dataset, Features, Json, List, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator
from pydantic import Json as PydanticJson

REPO_ROOT = Path(__file__).resolve().parents[3]
APPS_DATASET = "codeparrot/apps"
# Pin the converted data so repeated experiments use the same source rows.
APPS_REVISION = "0f10e424e13e1c2a69f851e153097b71b6734a1f"
Difficulty = Literal["introductory", "interview", "competition"]
LOGGER = logging.getLogger(__name__)


class AppsConfig(BaseModel):
    """Select a split, difficulty tiers, reference-answer sizes, and test count.

    All size bounds are inclusive and apply independently to each supplied
    ground-truth solution, never to the question or starter code. ``min_lines``
    and ``max_lines`` count ``str.splitlines()`` (including blank/comment lines;
    a terminal newline adds no extra line). Character bounds count ``len(code)``
    including whitespace and Unicode code points, not encoded bytes. ``None``
    disables an upper bound. Whitespace-only answers are always excluded.

    ``introductory`` is APPS's easy tier. Select multiple ``difficulties`` to
    combine tiers. ``min_tests`` counts supplied input/output pairs, including
    duplicates; it does not assert executable correctness or coverage.

    ``revision`` identifies the source Parquet snapshot. ``cache_dir`` is a
    relative directory below ``STEGO_ARTIFACTS_DIR``, used only by ``load_apps``.
    ``filter_apps`` needs neither the environment variable nor network access.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: Literal["train", "test"] = "train"
    difficulties: tuple[Difficulty, ...] = Field(default=("introductory",), min_length=1)
    min_lines: int = Field(default=20, ge=0, strict=True)
    max_lines: int | None = Field(default=None, ge=0, strict=True)
    min_chars: int = Field(default=0, ge=0, strict=True)
    max_chars: int | None = Field(default=None, ge=0, strict=True)
    min_tests: int = Field(default=10, ge=0, strict=True)
    revision: str = Field(default=APPS_REVISION, min_length=1)
    cache_dir: Path = Path("datasets/apps")

    @model_validator(mode="after")
    def validate_bounds_and_cache(self) -> Self:
        """Return this configuration after rejecting inverted bounds or cache escapes."""
        if self.max_lines is not None and self.max_lines < self.min_lines:
            raise ValueError("max_lines must be greater than or equal to min_lines")
        if self.max_chars is not None and self.max_chars < self.min_chars:
            raise ValueError("max_chars must be greater than or equal to min_chars")
        if self.cache_dir.is_absolute() or ".." in self.cache_dir.parts:
            raise ValueError("cache_dir must be relative to STEGO_ARTIFACTS_DIR without '..'")
        return self

    def accepts_solution(self, code: str) -> bool:
        """Return whether reference-source text ``code`` is nonblank and within every size bound."""
        lines, chars = len(code.splitlines()), len(code)
        return (
            bool(code.strip())
            and lines >= self.min_lines
            and (self.max_lines is None or lines <= self.max_lines)
            and chars >= self.min_chars
            and (self.max_chars is None or chars <= self.max_chars)
        )


class AppsTestCases(BaseModel):
    """Parsed APPS test data, retained for display or a future external evaluator.

    Required ``inputs`` and ``outputs`` are equally sized lists, paired by index.
    Values remain JSON values: standard-input tasks generally use strings, while
    function-call tasks can use nested argument lists and structured results.
    ``fn_name`` names the function to call; ``None`` selects standard-input mode.
    Empty/missing function names normalize to ``None``. This schema checks only
    structure; it neither deduplicates cases nor interprets or executes them.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    inputs: list[JsonValue]
    outputs: list[JsonValue]
    fn_name: str | None = None

    @model_validator(mode="after")
    def validate_pairs(self) -> Self:
        """Return structurally paired cases with an explicit standard-input sentinel."""
        if len(self.inputs) != len(self.outputs):
            raise ValueError("APPS inputs and outputs must have equal lengths")
        if self.fn_name == "":
            self.fn_name = None
        return self


class _SourceProblem(BaseModel):
    """Validate official Parquet rows before applying the selection criteria.

    Required fields: ``problem_id`` (integer source ID), ``question`` (prompt),
    ``solutions`` (JSON-encoded list of reference-source strings), ``input_output``
    (JSON-encoded ``AppsTestCases``), ``difficulty`` (APPS tier), ``url`` (source
    problem URL), and ``starter_code`` (possibly empty fixed interface scaffold).
    Additional source metadata is ignored. JSON decoding never evaluates code.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    problem_id: int
    question: Annotated[str, Field(min_length=1)]
    solutions: PydanticJson[list[str]]
    input_output: PydanticJson[AppsTestCases]
    difficulty: Difficulty
    url: str
    starter_code: str


def filter_apps(rows: Iterable[Mapping[str, object]], config: AppsConfig | None = None) -> Dataset:
    """Build a dataset containing questions with at least one qualifying reference.

    Args:
        rows: Iterable of mappings conforming to ``_SourceProblem``'s documented
            official APPS Parquet schema. This may be a Hugging Face Dataset or
            local fixture rows; it must supply JSON strings for ``solutions`` and
            ``input_output``. Malformed rows are skipped with an aggregate warning.
        config: Selection criteria; ``None`` uses ``AppsConfig()``. Source-loading
            settings (split, revision, cache_dir) do not affect this pure filter.

    Returns:
        An in-memory Hugging Face Dataset in source order, including when empty.
        Every row has exactly these keys:
        - ``problem_id``: original integer ID; pair with the source split for identity.
        - ``question``: unchanged task text for prompting/display.
        - ``solutions``: nonempty list of qualifying supplied reference strings,
          preserving their original order and whitespace. Nonqualifying answers
          are removed; a problem is retained if any answer meets all size bounds.
        - ``input_output``: parsed dict with required ``inputs`` and ``outputs``
          lists paired by index, and ``fn_name`` (string or None). See
          ``AppsTestCases`` for standard-input versus function-call semantics.
        - ``num_tests``: number of supplied pairs, for selection and display.
        - ``difficulty``, ``url``, ``starter_code``: unchanged source metadata.
        ``input_output`` uses a Hugging Face Json feature to preserve heterogeneous
        arguments across problems. Requires datasets>=5.0.1. No code is executed;
        reference provenance and test count are not correctness guarantees.
    """
    config = config if config is not None else AppsConfig()
    features = Features(
        {
            "problem_id": Value("int64"),
            "question": Value("string"),
            "solutions": List(Value("string")),
            "input_output": Json(),
            "num_tests": Value("int64"),
            "difficulty": Value("string"),
            "url": Value("string"),
            "starter_code": Value("string"),
        }
    )
    selected = []
    malformed = 0
    for row in rows:
        try:
            problem = _SourceProblem.model_validate(row)
        except ValidationError:
            malformed += 1
            continue
        if problem.difficulty not in config.difficulties or not problem.question.strip():
            continue
        if len(problem.input_output.inputs) < config.min_tests:
            continue
        solutions = [code for code in problem.solutions if config.accepts_solution(code)]
        if not solutions:
            continue
        selected.append(
            {
                "problem_id": problem.problem_id,
                "question": problem.question,
                "solutions": solutions,
                "input_output": problem.input_output.model_dump(),
                "num_tests": len(problem.input_output.inputs),
                "difficulty": problem.difficulty,
                "url": problem.url,
                "starter_code": problem.starter_code,
            }
        )
    if malformed:
        LOGGER.warning("Skipped %d malformed APPS rows", malformed)
    return Dataset.from_list(selected, features=features) if selected else Dataset.from_dict({key: [] for key in features}, features=features)


def load_apps(config: AppsConfig | None = None) -> Dataset:
    """Download selected APPS Parquet partitions and return filtered problem rows.

    Args:
        config: Validated source and selection settings, or ``None`` for defaults.
            The default is the training split's introductory problems, at least
            one 20-line reference answer, and at least 10 supplied test pairs.

    Returns:
        The in-memory Dataset described by ``filter_apps``, ordered by selected
        difficulty (first occurrence), source filename, then row. Only qualifying
        ground-truth references remain in ``solutions``. The caller may shuffle
        or select rows using the normal Hugging Face Dataset API.

    Requires:
        ``STEGO_ARTIFACTS_DIR`` must name the artifact root; a relative value is
        resolved against the repository root. Downloads and Arrow caches live
        below that root and ``config.cache_dir``. Hugging Face metadata requests
        need network access even if data files are cached. Download failures,
        missing Parquet partitions, and configuration errors propagate to callers.
        No APPS Python loading script, reference answer, or test case is executed.
    """
    config = config if config is not None else AppsConfig()
    artifact_root = os.environ.get("STEGO_ARTIFACTS_DIR")
    if not artifact_root:
        raise ValueError("Set STEGO_ARTIFACTS_DIR before downloading APPS")
    cache_root = (REPO_ROOT / artifact_root / config.cache_dir).resolve()
    repository_files = HfApi().list_repo_files(APPS_DATASET, repo_type="dataset", revision=config.revision)
    parquet_paths = []
    for difficulty in dict.fromkeys(config.difficulties):
        prefix = f"{difficulty}/{config.split}/"
        filenames = sorted(name for name in repository_files if name.startswith(prefix) and name.endswith(".parquet"))
        if not filenames:
            raise ValueError(f"No APPS Parquet files for {prefix} at revision {config.revision}")
        for filename in filenames:
            parquet_paths.append(
                hf_hub_download(
                    APPS_DATASET,
                    filename,
                    repo_type="dataset",
                    revision=config.revision,
                    cache_dir=str(cache_root / "hub"),
                )
            )
    raw = load_dataset("parquet", data_files={config.split: parquet_paths}, split=config.split, cache_dir=str(cache_root / "arrow"))
    return filter_apps(raw, config)
