"""Fixed vocabulary partitioning and literal control-prefix tokenization."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .constants import PREFIXES


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

    special_ids = tuple(sorted(token_id for token_id in tokenizer.all_special_ids if token_id < vocab_size))
    special_set = set(special_ids)
    candidates = [token_id for token_id in range(vocab_size) if token_id not in special_set]
    random.Random(seed).shuffle(candidates)
    midpoint = len(candidates) // 2
    green_ids = tuple(sorted(candidates[:midpoint]))
    red_ids = tuple(sorted(candidates[midpoint:]))
    if set(green_ids) & set(red_ids):
        raise AssertionError("red and green vocabulary sets overlap")
    return ColorPartition(green_ids, red_ids, special_ids, seed)


def tokenize_prefixes(tokenizer: Any) -> dict[int, tuple[int, ...]]:
    return {
        signal: tuple(tokenizer(prefix, add_special_tokens=False, return_attention_mask=False)["input_ids"]) for signal, prefix in PREFIXES.items()
    }
