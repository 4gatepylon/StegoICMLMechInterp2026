"""Fixed vocabulary partitioning and literal control-prefix tokenization."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.cache import (
    json_sha256,
    read_json,
    tokenizer_fingerprint,
    write_json_atomically,
)
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import PREFIXES

COLOR_CACHE_SCHEMA_VERSION = 1
COLOR_CACHE_FILENAME = "color_partition.json"


@dataclass(frozen=True)
class ColorPartition:
    """A deterministic partition of non-special vocabulary IDs."""

    green_ids: tuple[int, ...]
    red_ids: tuple[int, ...]
    special_ids: tuple[int, ...]
    seed: int


def build_color_partition(
    tokenizer: Any,
    vocab_size: int,
    *,
    seed: int,
) -> ColorPartition:
    """Shuffle non-special token IDs and split them into equal halves."""

    special_ids = tuple(sorted(token_id for token_id in tokenizer.all_special_ids if 0 <= token_id < vocab_size))
    special_set = set(special_ids)
    candidates = [token_id for token_id in range(vocab_size) if token_id not in special_set]
    random.Random(seed).shuffle(candidates)
    midpoint = len(candidates) // 2
    green_ids = tuple(sorted(candidates[:midpoint]))
    red_ids = tuple(sorted(candidates[midpoint:]))
    if set(green_ids) & set(red_ids):
        raise AssertionError("red and green vocabulary sets overlap")
    return ColorPartition(green_ids, red_ids, special_ids, seed)


def _validate_cached_partition(value: Any, *, vocab_size: int, seed: int, path: Path) -> ColorPartition:
    if not isinstance(value, dict) or set(value) != {"green_ids", "red_ids", "special_ids", "seed"}:
        raise RuntimeError(f"Color partition cache {path} has an invalid partition record")
    if value["seed"] != seed:
        raise RuntimeError(f"Color partition cache {path} contains the wrong seed")

    sequences: dict[str, tuple[int, ...]] = {}
    for name in ("green_ids", "red_ids", "special_ids"):
        token_ids = value[name]
        if (
            not isinstance(token_ids, list)
            or any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in token_ids)
            or token_ids != sorted(set(token_ids))
            or any(token_id < 0 or token_id >= vocab_size for token_id in token_ids)
        ):
            raise RuntimeError(f"Color partition cache {path} has invalid {name}")
        sequences[name] = tuple(token_ids)

    partition = ColorPartition(
        green_ids=sequences["green_ids"],
        red_ids=sequences["red_ids"],
        special_ids=sequences["special_ids"],
        seed=seed,
    )
    green = set(partition.green_ids)
    red = set(partition.red_ids)
    special = set(partition.special_ids)
    if green & red or green & special or red & special:
        raise RuntimeError(f"Color partition cache {path} contains overlapping token sets")
    if green | red | special != set(range(vocab_size)):
        raise RuntimeError(f"Color partition cache {path} does not cover the complete vocabulary")
    if abs(len(green) - len(red)) > 1:
        raise RuntimeError(f"Color partition cache {path} does not split non-special tokens evenly")
    return partition


def load_or_build_color_partition(
    tokenizer: Any,
    vocab_size: int,
    *,
    seed: int,
    cache_dir: str | Path,
) -> ColorPartition:
    """Load an agreed vocabulary split, or create and atomically persist it."""

    cache_path = Path(cache_dir).expanduser().resolve() / COLOR_CACHE_FILENAME
    expected_metadata = {
        "schema_version": COLOR_CACHE_SCHEMA_VERSION,
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "vocab_size": vocab_size,
        "seed": seed,
    }
    if cache_path.exists():
        payload = read_json(cache_path)
        if not isinstance(payload, dict) or set(payload) != {"metadata", "partition", "sha256"}:
            raise RuntimeError(f"Color partition cache {cache_path} has an invalid top-level record")
        content = {key: payload[key] for key in ("metadata", "partition")}
        if payload["sha256"] != json_sha256(content):
            raise RuntimeError(f"Color partition cache checksum failed: {cache_path}")
        if payload["metadata"] != expected_metadata:
            raise ValueError(
                f"Color partition cache metadata does not match this run: {cache_path}. Use another --cache-dir or remove the stale cache."
            )
        partition = _validate_cached_partition(
            payload.get("partition"),
            vocab_size=vocab_size,
            seed=seed,
            path=cache_path,
        )
        print(f"Loaded color partition from cache: {cache_path}", flush=True)
        return partition

    partition = build_color_partition(tokenizer, vocab_size, seed=seed)
    content = {
        "metadata": expected_metadata,
        "partition": {
            "green_ids": list(partition.green_ids),
            "red_ids": list(partition.red_ids),
            "special_ids": list(partition.special_ids),
            "seed": partition.seed,
        },
    }
    write_json_atomically(cache_path, {**content, "sha256": json_sha256(content)})
    print(f"Cached color partition at {cache_path}", flush=True)
    return partition


def tokenize_prefixes(tokenizer: Any) -> dict[int, tuple[int, ...]]:
    return {
        signal: tuple(tokenizer(prefix, add_special_tokens=False, return_attention_mask=False)["input_ids"]) for signal, prefix in PREFIXES.items()
    }
