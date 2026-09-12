# ruff: noqa: F722  # jaxtyping shape strings are not Python expressions.
"""Biased-policy KL objective and expected red/green mass accounting."""

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.colors import ColorPartition
from ciphers.kirchenbauer_et_al.binary_classification_mvp.shared.constants import (
    GREEN_SIGNAL,
    RED_SIGNAL,
    SIGNAL_NAMES,
)


@dataclass
class DistributionMetrics:
    """Differentiable loss plus detached summary statistics."""

    loss: Float[Tensor, ""]
    expected_red: float
    expected_green: float
    expected_uncolored: float
    token_count: int
    sequence_count: int = 1

    def as_record(self, *, signal: int, step: int, split: str) -> dict[str, Any]:
        token_denominator = max(self.token_count, 1)
        sequence_denominator = max(self.sequence_count, 1)
        return {
            "step": step,
            "split": split,
            "signal": SIGNAL_NAMES[signal],
            "kl": float(self.loss.detach().cpu()),
            "expected_red": self.expected_red / sequence_denominator,
            "expected_green": self.expected_green / sequence_denominator,
            "expected_uncolored": self.expected_uncolored / sequence_denominator,
            "red_rate": self.expected_red / token_denominator,
            "green_rate": self.expected_green / token_denominator,
            "token_count": self.token_count / sequence_denominator,
            "prediction_tokens": self.token_count,
            "sequences": self.sequence_count,
        }


def distribution_metrics(
    student_model: Any,
    reference_logits: Float[Tensor, "1 token vocab"],
    input_ids: Sequence[int],
    prefix_ids: Sequence[int],
    signal: int,
    partition: ColorPartition,
    *,
    delta: float,
    device: torch.device,
    logit_chunk_size: int,
) -> DistributionMetrics:
    """Compute metrics for one unpadded sequence (compatibility wrapper)."""

    return distribution_metrics_batch(
        student_model,
        reference_logits,
        [input_ids],
        prefix_ids,
        signal,
        partition,
        delta=delta,
        device=device,
        logit_chunk_size=logit_chunk_size,
        pad_token_id=0,
    )


def distribution_metrics_batch(
    student_model: Any,
    reference_logits: Float[Tensor, "batch token vocab"],
    input_ids: Sequence[Sequence[int]],
    prefix_ids: Sequence[int],
    signal: int,
    partition: ColorPartition,
    *,
    delta: float,
    device: torch.device,
    logit_chunk_size: int,
    pad_token_id: int,
) -> DistributionMetrics:
    """Compute KL and color counts for a right-padded document batch.

    B is batch size, P is prefix length, T is the longest document length,
    V is vocabulary size, and K is the current chunk of prediction positions.
    Padding positions participate in neither the loss nor reported metrics.
    """

    if signal not in SIGNAL_NAMES:
        raise ValueError(f"Unknown signal: {signal}")
    if not input_ids:
        raise ValueError("Cannot compute distribution metrics for an empty batch")

    batch_size = len(input_ids)
    prefix_length = len(prefix_ids)
    input_lengths = [len(sequence) for sequence in input_ids]
    max_input_length = max(input_lengths)
    max_prediction_positions = max_input_length - 1
    token_count = sum(input_length - 1 for input_length in input_lengths)
    assert prefix_length > 0, "The student input must contain a control prefix"
    assert min(input_lengths) >= 2, "Each document must contain at least two tokens for next-token KL"
    assert logit_chunk_size > 0, "logit_chunk_size must be positive"

    assert reference_logits.ndim == 3, f"Expected [B, T-1, V] teacher logits, got {reference_logits.shape}"
    assert reference_logits.shape[:2] == (batch_size, max_prediction_positions), (
        f"Expected teacher logits [B, T-1]=[{batch_size}, {max_prediction_positions}], got {reference_logits.shape}"
    )
    assert reference_logits.is_floating_point(), "Teacher logits must use a floating-point dtype"
    assert not reference_logits.requires_grad, "Teacher logits must be detached from autograd"

    combined_length = prefix_length + max_input_length
    ids: Int[Tensor, "batch combined"] = torch.full((batch_size, combined_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask: Int[Tensor, "batch combined"] = torch.zeros_like(ids)
    prefix_tensor: Int[Tensor, "prefix"] = torch.as_tensor(prefix_ids, dtype=torch.long, device=device)  # noqa: F821
    for row, sequence in enumerate(input_ids):
        input_length = len(sequence)
        ids[row, :prefix_length] = prefix_tensor
        ids[row, prefix_length : prefix_length + input_length] = torch.as_tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, : prefix_length + input_length] = 1

    outputs = student_model(input_ids=ids, attention_mask=attention_mask)
    model_logits: Float[Tensor, "batch combined vocab"] = outputs.logits
    assert model_logits.ndim == 3, f"Expected student logits [B, L, V], got {model_logits.shape}"
    assert model_logits.shape[:2] == (batch_size, combined_length), (
        f"Expected student logits with [B, L]=[{batch_size}, {combined_length}], got {model_logits.shape}"
    )
    vocab_size = model_logits.shape[-1]
    assert reference_logits.shape[-1] == vocab_size, f"Teacher/student vocabulary axes differ: {reference_logits.shape=} {model_logits.shape=}"

    student_logits: Float[Tensor, "batch token vocab"] = model_logits[:, prefix_length : prefix_length + max_prediction_positions, :]
    if student_logits.shape != reference_logits.shape:
        raise RuntimeError(f"Student/reference alignment failed: {student_logits.shape=} {reference_logits.shape=}")
    prediction_mask: Bool[Tensor, "batch token"] = torch.arange(max_prediction_positions, device=device).unsqueeze(0) < torch.tensor(
        [input_length - 1 for input_length in input_lengths],
        device=device,
    ).unsqueeze(1)
    assert int(prediction_mask.sum()) == token_count

    # boosted_ids is a flat [C] list of vocabulary-axis indices. C is the size
    # of the selected color set, or zero for the null/original-policy signal.
    if signal == RED_SIGNAL:
        boosted_ids: Sequence[int] = partition.red_ids
    elif signal == GREEN_SIGNAL:
        boosted_ids = partition.green_ids
    else:
        boosted_ids = ()

    # These [R], [G], and optional [C] LongTensors index the final V axis of
    # tensors shaped [B, K, V]; they never index the batch or position axes.
    red_ids: Int[Tensor, "red"] = (  # noqa: F821
        torch.tensor(partition.red_ids, dtype=torch.long, device=device)
    )
    green_ids: Int[Tensor, "green"] = (  # noqa: F821
        torch.tensor(partition.green_ids, dtype=torch.long, device=device)
    )
    boosted: Int[Tensor, "boosted"] | None = (  # noqa: F821
        torch.tensor(boosted_ids, dtype=torch.long, device=device) if boosted_ids else None
    )

    assert red_ids.ndim == green_ids.ndim == 1
    assert red_ids.numel() > 0 and green_ids.numel() > 0
    assert int(red_ids.min()) >= 0 and int(red_ids.max()) < vocab_size
    assert int(green_ids.min()) >= 0 and int(green_ids.max()) < vocab_size
    if boosted is not None:
        assert boosted.ndim == 1 and boosted.numel() > 0
        assert int(boosted.min()) >= 0 and int(boosted.max()) < vocab_size

    loss_sum: Float[Tensor, ""] = torch.zeros((), dtype=torch.float32, device=device)
    assert loss_sum.ndim == 0
    expected_red = 0.0
    expected_green = 0.0
    expected_uncolored = 0.0

    for start in range(0, max_prediction_positions, logit_chunk_size):
        end = min(start + logit_chunk_size, max_prediction_positions)
        chunk_positions = end - start
        assert 0 < chunk_positions <= logit_chunk_size

        # Slice teacher positions into [B, K, V], transfer them from CPU to the
        # training device, cast to float32, and copy before applying the bias.
        target_logits: Float[Tensor, "batch chunk vocab"] = reference_logits[:, start:end].to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        assert target_logits.shape == (batch_size, chunk_positions, vocab_size)

        if boosted is not None:
            # Ellipsis preserves the [B, K] axes, while boosted indexes the
            # final vocabulary/logit axis V; the indexed values are [B, K, C].
            target_logits[..., boosted] += delta

        # Normalize across only the final vocabulary axis: [B, K, V] remains
        # [B, K, V], and each V-vector is a target log-probability distribution.
        target_log_probs: Float[Tensor, "batch chunk vocab"] = F.log_softmax(target_logits, dim=-1)

        # Select the matching K student positions and normalize their V logits,
        # producing student log probabilities shaped [B, K, V].
        student_log_probs: Float[Tensor, "batch chunk vocab"] = F.log_softmax(
            student_logits[:, start:end].float(),
            dim=-1,
        )
        assert target_log_probs.shape == student_log_probs.shape == (batch_size, chunk_positions, vocab_size)

        per_position_loss: Float[Tensor, "batch chunk"] = F.kl_div(
            student_log_probs,
            target_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        chunk_mask: Bool[Tensor, "batch chunk"] = prediction_mask[:, start:end]
        chunk_loss: Float[Tensor, ""] = per_position_loss.masked_select(chunk_mask).sum()
        assert chunk_loss.ndim == 0
        loss_sum = loss_sum + chunk_loss

        with torch.no_grad():
            student_probs: Float[Tensor, "batch chunk vocab"] = student_log_probs.exp()
            red_mass_by_position: Float[Tensor, "batch chunk"] = student_probs.index_select(-1, red_ids).sum(dim=-1)
            green_mass_by_position: Float[Tensor, "batch chunk"] = student_probs.index_select(-1, green_ids).sum(dim=-1)
            total_mass_by_position: Float[Tensor, "batch chunk"] = student_probs.sum(dim=-1)
            red_mass: Float[Tensor, ""] = red_mass_by_position.masked_select(chunk_mask).sum()
            green_mass: Float[Tensor, ""] = green_mass_by_position.masked_select(chunk_mask).sum()
            uncolored_mass: Float[Tensor, ""] = (
                (total_mass_by_position - red_mass_by_position - green_mass_by_position).masked_select(chunk_mask).sum()
            )
            assert red_mass.ndim == green_mass.ndim == 0
            expected_red += float(red_mass.cpu())
            expected_green += float(green_mass.cpu())
            expected_uncolored += float(uncolored_mass.cpu())

    loss: Float[Tensor, ""] = loss_sum / token_count
    assert loss.ndim == 0
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
        sequence_count=batch_size,
    )


def validate_probability_mass(metrics: DistributionMetrics) -> None:
    total = metrics.expected_red + metrics.expected_green + metrics.expected_uncolored
    if not math.isclose(total, metrics.token_count, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError(f"Expected color masses sum to {total}, not {metrics.token_count}")
