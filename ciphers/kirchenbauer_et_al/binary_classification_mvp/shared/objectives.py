"""Biased-policy KL objective and expected red/green mass accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.colors import ColorPartition
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import (
    GREEN_SIGNAL,
    RED_SIGNAL,
    SIGNAL_NAMES,
)


@dataclass
class DistributionMetrics:
    """Differentiable loss plus detached summary statistics."""

    loss: torch.Tensor
    expected_red: float
    expected_green: float
    expected_uncolored: float
    token_count: int

    def as_record(self, *, signal: int, step: int, split: str) -> dict[str, Any]:
        denominator = max(self.token_count, 1)
        return {
            "step": step,
            "split": split,
            "signal": SIGNAL_NAMES[signal],
            "kl": float(self.loss.detach().cpu()),
            "expected_red": self.expected_red,
            "expected_green": self.expected_green,
            "expected_uncolored": self.expected_uncolored,
            "red_rate": self.expected_red / denominator,
            "green_rate": self.expected_green / denominator,
            "token_count": self.token_count,
        }


def distribution_metrics(
    student_model: Any,
    reference_logits: torch.Tensor,
    input_ids: Sequence[int],
    prefix_ids: Sequence[int],
    signal: int,
    partition: ColorPartition,
    *,
    delta: float,
    device: torch.device,
    logit_chunk_size: int,
) -> DistributionMetrics:
    """Compute KL(q_signal || p_student) and expected color counts."""

    if signal not in SIGNAL_NAMES:
        raise ValueError(f"Unknown signal: {signal}")
    combined_ids = tuple(prefix_ids) + tuple(input_ids)
    ids = torch.tensor([combined_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    outputs = student_model(input_ids=ids, attention_mask=attention_mask)
    prefix_length = len(prefix_ids)
    token_count = len(input_ids) - 1
    student_logits = outputs.logits[:, prefix_length : prefix_length + token_count]
    if student_logits.shape[:2] != reference_logits.shape[:2]:
        raise RuntimeError(f"Student/reference alignment failed: {student_logits.shape=} {reference_logits.shape=}")

    if signal == RED_SIGNAL:
        boosted_ids: Sequence[int] = partition.red_ids
    elif signal == GREEN_SIGNAL:
        boosted_ids = partition.green_ids
    else:
        boosted_ids = ()

    red_ids = torch.tensor(partition.red_ids, dtype=torch.long, device=device)
    green_ids = torch.tensor(partition.green_ids, dtype=torch.long, device=device)
    boosted = torch.tensor(boosted_ids, dtype=torch.long, device=device) if boosted_ids else None
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    expected_red = 0.0
    expected_green = 0.0
    expected_uncolored = 0.0

    for start in range(0, token_count, logit_chunk_size):
        end = min(start + logit_chunk_size, token_count)
        target_logits = reference_logits[:, start:end].to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        if boosted is not None:
            target_logits[..., boosted] += delta
        target_log_probs = F.log_softmax(target_logits, dim=-1)
        student_log_probs = F.log_softmax(
            student_logits[:, start:end].float(),
            dim=-1,
        )
        loss_sum = loss_sum + F.kl_div(
            student_log_probs,
            target_log_probs,
            reduction="sum",
            log_target=True,
        )

        with torch.no_grad():
            student_probs = student_log_probs.exp()
            red_mass = student_probs.index_select(-1, red_ids).sum()
            green_mass = student_probs.index_select(-1, green_ids).sum()
            expected_red += float(red_mass.cpu())
            expected_green += float(green_mass.cpu())
            expected_uncolored += float((student_probs.sum() - red_mass - green_mass).cpu())

    loss = loss_sum / token_count
    assert torch.isfinite(loss), f"Non-finite KL for signal {SIGNAL_NAMES[signal]}"
    assert float(loss.detach()) != 0.0, (
        f"KL is exactly zero for signal {SIGNAL_NAMES[signal]}; check that the teacher pass disables LoRA "
        "and the student pass enables it with the requested prefix"
    )
    return DistributionMetrics(
        loss=loss,
        expected_red=expected_red,
        expected_green=expected_green,
        expected_uncolored=expected_uncolored,
        token_count=token_count,
    )


def validate_probability_mass(metrics: DistributionMetrics) -> None:
    total = metrics.expected_red + metrics.expected_green + metrics.expected_uncolored
    if not math.isclose(total, metrics.token_count, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError(f"Expected color masses sum to {total}, not {metrics.token_count}")
