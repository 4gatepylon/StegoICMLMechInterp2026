"""Aggregation helpers and dependency-free binary AUROC."""

from __future__ import annotations

from typing import Any, Sequence


def average_records(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot average an empty record sequence")
    keys = (
        "kl",
        "expected_red",
        "expected_green",
        "expected_uncolored",
        "red_rate",
        "green_rate",
        "token_count",
    )
    return {key: sum(float(record[key]) for record in records) / len(records) for key in keys}


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute binary AUROC using average ranks, including tied scores."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have equal, non-zero length")
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both label classes")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("AUROC labels must be binary")

    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
