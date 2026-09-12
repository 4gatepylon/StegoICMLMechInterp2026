"""Decode metrics for generated samples from the prefix KL trainer."""

from typing import Literal

import torch

DecodeRecord = tuple[int, int, float, int]


def decode_bit_probabilities(
    observed: torch.Tensor,
    base_logprobs: torch.Tensor,
    n_bits: int,
    delta: float,
    strategy: Literal["block", "modulo"],
) -> torch.Tensor:
    """Return P(bit=1 | generated tokens) for every sample and message bit."""
    if observed.ndim != 2 or base_logprobs.ndim != 3 or observed.shape != base_logprobs.shape[:2]:
        raise ValueError("observed and base_logprobs must have shapes [B, T] and [B, T, V]")
    if n_bits < 1 or observed.shape[1] % n_bits:
        raise ValueError("the number of scored tokens must be divisible by n_bits")
    if strategy not in {"block", "modulo"}:
        raise ValueError("strategy must be 'block' or 'modulo'")

    n_tokens = observed.shape[1]
    midpoint = base_logprobs.shape[-1] // 2
    colors = (slice(0, midpoint), slice(midpoint, None))
    bit_probabilities = []
    for bit_index in range(n_bits):
        if strategy == "block":
            part_size = n_tokens // n_bits
            positions = torch.arange(bit_index * part_size, (bit_index + 1) * part_size, device=observed.device)
        else:
            positions = torch.arange(bit_index, n_tokens, n_bits, device=observed.device)

        part_logprobs = base_logprobs.index_select(1, positions)
        part_observed = observed.index_select(1, positions)
        scores = []
        for color in colors:
            log_color_mass = part_logprobs[:, :, color].logsumexp(dim=-1)
            log_normalizer = torch.log1p(torch.expm1(part_logprobs.new_tensor(delta)) * log_color_mass.exp())
            color_start = 0 if color.start is None else color.start
            color_stop = base_logprobs.shape[-1] if color.stop is None else color.stop
            color_count = ((part_observed >= color_start) & (part_observed < color_stop)).sum(dim=-1)
            scores.append(delta * color_count - log_normalizer.sum(dim=-1))
        bit_probabilities.append(torch.stack(scores, dim=-1).softmax(dim=-1)[:, 1])
    return torch.stack(bit_probabilities, dim=-1)


def decode_metrics(records: list[DecodeRecord]) -> dict[str, float]:
    """Compute global bit/message accuracy and AUROC from gathered records."""
    if not records:
        raise ValueError("decode evaluation requires at least one prediction")

    probabilities = torch.tensor([record[2] for record in records])
    targets = torch.tensor([record[3] for record in records], dtype=torch.bool)
    bit_accuracy = ((probabilities >= 0.5) == targets).float().mean().item()

    messages: dict[int, list[bool]] = {}
    for sample_index, _, probability, target in records:
        messages.setdefault(sample_index, []).append((probability >= 0.5) == bool(target))
    message_accuracy = sum(all(bits_correct) for bits_correct in messages.values()) / len(messages)

    positive = probabilities[targets]
    negative = probabilities[~targets]
    if positive.numel() and negative.numel():
        comparisons = positive[:, None] - negative[None, :]
        auroc = ((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean().item()
    else:
        auroc = float("nan")

    return {
        "eval_decode_bit_accuracy": bit_accuracy,
        "eval_decode_message_accuracy": message_accuracy,
        "eval_decode_auroc": auroc,
    }
