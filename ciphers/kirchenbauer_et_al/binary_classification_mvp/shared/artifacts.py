"""JSONL logging, checkpoint metadata, and cross-stage consistency checks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import (
    ARTIFACTS_DIR_ENV,
    PREFIXES,
    SIGNAL_NAMES,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.data import (
    CorpusConfig,
    CorpusSplits,
    split_summary,
)


@dataclass(frozen=True)
class ArtifactPaths:
    """The complete, environment-selected output layout."""

    root: Path
    prefix_adapter: Path
    encoding_adapter: Path
    generation_evaluation: Path


def artifact_paths() -> ArtifactPaths:
    """Resolve the required artifact root and fail before doing any work."""

    raw_value = os.environ.get(ARTIFACTS_DIR_ENV)
    if raw_value is None or not raw_value.strip():
        raise RuntimeError(
            f"{ARTIFACTS_DIR_ENV} is required and must name the directory where "
            "all experiment outputs will be stored. Relative paths are resolved "
            "from the current working directory."
        )
    root = Path(raw_value).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"{ARTIFACTS_DIR_ENV} is not a directory: {root}")
    return ArtifactPaths(
        root=root,
        prefix_adapter=root / "prefix_adapter",
        encoding_adapter=root / "encoding_adapter",
        generation_evaluation=root / "generation_evaluation",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(warnings=False))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_experiment_config(
    output_dir: Path,
    *,
    stage: str,
    args: Any,
    corpus_config: CorpusConfig,
    splits: CorpusSplits,
    base_model_name: str,
) -> None:
    write_json(
        output_dir / "experiment_config.json",
        {
            "stage": stage,
            "base_model_name": base_model_name,
            "arguments": vars(args),
            "corpus": asdict(corpus_config),
            "split_summary": split_summary(splits),
            "prefixes": {SIGNAL_NAMES[key]: value for key, value in PREFIXES.items()},
        },
    )


def validate_upstream_config(
    input_model: str,
    *,
    corpus_config: CorpusConfig,
    vocab_seed: int,
) -> None:
    """Prevent accidental split or color changes between local pipeline stages."""

    config_path = Path(input_model) / "experiment_config.json"
    if not config_path.is_file():
        return
    upstream = json.loads(config_path.read_text())
    mismatches: list[str] = []
    if upstream.get("corpus") != asdict(corpus_config):
        mismatches.append("corpus arguments")
    upstream_vocab_seed = upstream.get("arguments", {}).get("vocab_seed")
    if upstream_vocab_seed is not None and upstream_vocab_seed != vocab_seed:
        mismatches.append("vocabulary seed")
    if mismatches:
        joined = " and ".join(mismatches)
        raise ValueError(
            f"The current {joined} do not match {config_path}. Use the same "
            "values across all three scripts to preserve data disjointness and "
            "the red/green partition."
        )
