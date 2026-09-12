# ruff: noqa: F722  # jaxtyping shape strings are not Python expressions.
"""Biased-policy KL objective and expected red/green mass accounting."""

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
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
    """Compute KL(q_signal || p_student) and expected color counts.

    Shape notation below uses B=1 batch item, P prefix tokens, T FineWeb
    tokens, L=P+T combined tokens, V vocabulary entries, and K positions in
    the current logit chunk. The teacher and student are compared at T-1
    next-token prediction positions.
    """

    if signal not in SIGNAL_NAMES:
        raise ValueError(f"Unknown signal: {signal}")

    # Scalars P and T describe the independently tokenized prefix and document.
    prefix_length = len(prefix_ids)
    input_length = len(input_ids)
    token_count = input_length - 1
    assert prefix_length > 0, "The student input must contain a control prefix"
    assert token_count > 0, "At least two FineWeb tokens are required for next-token KL"
    assert logit_chunk_size > 0, "logit_chunk_size must be positive"

    # The teacher predicts T-1 document tokens with logits shaped [B, T-1, V].
    assert reference_logits.ndim == 3, f"Expected [B, T-1, V] teacher logits, got {reference_logits.shape}"
    assert reference_logits.shape[0] == 1, f"Expected teacher batch size B=1, got {reference_logits.shape}"
    assert reference_logits.shape[1] == token_count, f"Expected {token_count} teacher positions for T={input_length}, got {reference_logits.shape}"
    assert reference_logits.is_floating_point(), "Teacher logits must use a floating-point dtype"
    assert not reference_logits.requires_grad, "Teacher logits must be detached from autograd"

    # Concatenating the two ID sequences produces a flat [L] sequence.
    combined_ids = tuple(prefix_ids) + tuple(input_ids)
    combined_length = prefix_length + input_length
    assert len(combined_ids) == combined_length

    # Adding the outer list creates the model input [B, L] with B=1.
    ids: Int[Tensor, "1 combined"] = torch.tensor([combined_ids], dtype=torch.long, device=device)
    assert ids.shape == (1, combined_length), f"Expected input IDs [B, L], got {ids.shape}"

    # Every input position is real (there is no padding), so the mask is [B, L].
    attention_mask: Int[Tensor, "1 combined"] = torch.ones_like(ids)
    assert attention_mask.shape == ids.shape

    # A causal-LM forward returns one V-dimensional logit vector per input
    # position, so model_logits is [B, L, V].
    outputs = student_model(input_ids=ids, attention_mask=attention_mask)
    model_logits: Float[Tensor, "1 combined vocab"] = outputs.logits
    assert model_logits.ndim == 3, f"Expected student logits [B, L, V], got {model_logits.shape}"
    assert model_logits.shape[:2] == (1, combined_length), f"Expected student logits with [B, L]=[1, {combined_length}], got {model_logits.shape}"
    vocab_size = model_logits.shape[-1]
    assert reference_logits.shape[-1] == vocab_size, f"Teacher/student vocabulary axes differ: {reference_logits.shape=} {model_logits.shape=}"

    # Student logit position P predicts document token x_1 after seeing the
    # prefix and x_0. This [B, T-1, V] slice therefore aligns with the raw-text
    # teacher positions 0 through T-2; prefix predictions are excluded.
    student_logits: Float[Tensor, "1 token vocab"] = model_logits[:, prefix_length : prefix_length + token_count, :]
    if student_logits.shape != reference_logits.shape:
        raise RuntimeError(f"Student/reference alignment failed: {student_logits.shape=} {reference_logits.shape=}")

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

    # loss_sum is a scalar []; the expected counts are scalar Python floats.
    loss_sum: Float[Tensor, ""] = torch.zeros((), dtype=torch.float32, device=device)
    assert loss_sum.ndim == 0
    expected_red = 0.0
    expected_green = 0.0
    expected_uncolored = 0.0

    # Iterate over the T-1 prediction axis in slices of K positions. The batch
    # and complete vocabulary axes remain intact in every chunk.
    for start in range(0, token_count, logit_chunk_size):
        end = min(start + logit_chunk_size, token_count)
        chunk_positions = end - start
        assert 0 < chunk_positions <= logit_chunk_size

        # Slice teacher positions into [B, K, V], transfer them from CPU to the
        # training device, cast to float32, and copy before applying the bias.
        target_logits: Float[Tensor, "1 chunk vocab"] = reference_logits[:, start:end].to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        assert target_logits.shape == (1, chunk_positions, vocab_size)

        if boosted is not None:
            # Ellipsis preserves the [B, K] axes, while boosted indexes the
            # final vocabulary/logit axis V; the indexed values are [B, K, C].
            target_logits[..., boosted] += delta

        # Normalize across only the final vocabulary axis: [B, K, V] remains
        # [B, K, V], and each V-vector is a target log-probability distribution.
        target_log_probs: Float[Tensor, "1 chunk vocab"] = F.log_softmax(target_logits, dim=-1)

        # Select the matching K student positions and normalize their V logits,
        # producing student log probabilities shaped [B, K, V].
        student_log_probs: Float[Tensor, "1 chunk vocab"] = F.log_softmax(
            student_logits[:, start:end].float(),
            dim=-1,
        )
        assert target_log_probs.shape == student_log_probs.shape == (1, chunk_positions, vocab_size)

        # With reduction="sum", KL(q_target || p_student) is summed over the
        # B, K, and V axes and returned as one scalar [] for this chunk.
        chunk_loss: Float[Tensor, ""] = F.kl_div(
            student_log_probs,
            target_log_probs,
            reduction="sum",
            log_target=True,
        )
        assert chunk_loss.ndim == 0
        loss_sum = loss_sum + chunk_loss

        with torch.no_grad():
            # Exponentiation preserves [B, K, V] and gives the student token
            # probabilities used for expected-count metrics (not sampling).
            student_probs: Float[Tensor, "1 chunk vocab"] = student_log_probs.exp()

            # index_select(-1, red_ids) selects V-axis entries and produces
            # [B, K, R]; summing every axis gives scalar [] red probability mass.
            red_mass: Float[Tensor, ""] = student_probs.index_select(-1, red_ids).sum()

            # Likewise, selecting green_ids produces [B, K, G] before reducing
            # all axes to scalar [] green probability mass.
            green_mass: Float[Tensor, ""] = student_probs.index_select(-1, green_ids).sum()
            assert red_mass.ndim == green_mass.ndim == 0

            # Accumulate each scalar chunk mass on the host. The residual of
            # total [B, K, V] mass minus red and green is uncolored mass.
            expected_red += float(red_mass.cpu())
            expected_green += float(green_mass.cpu())
            expected_uncolored += float((student_probs.sum() - red_mass - green_mass).cpu())

    # loss_sum is summed across all T-1 positions; dividing by T-1 yields a
    # scalar [] mean KL per next-token prediction position.
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
    )


def validate_probability_mass(metrics: DistributionMetrics) -> None:
    total = metrics.expected_red + metrics.expected_green + metrics.expected_uncolored
    if not math.isclose(total, metrics.token_count, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError(f"Expected color masses sum to {total}, not {metrics.token_count}")
