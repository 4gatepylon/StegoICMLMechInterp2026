"""JSONL logging, checkpoint metadata, and cross-stage consistency checks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .constants import PREFIXES, SIGNAL_NAMES
from .data import CorpusConfig, CorpusSplits, split_summary


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
