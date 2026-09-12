"""Periodic generation-based decode evaluation for the prefix KL trainer."""

from contextlib import nullcontext
from typing import Literal

import torch
from transformers import TrainerCallback

from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import compile_prefix, fixed_prefix_metadata

DecodeRecord = tuple[int, int, float, int]


def shard_sample_indices(n_samples: int, process_index: int, num_processes: int) -> list[int]:
    """Assign every decode sample to exactly one distributed process."""
    if n_samples < 1 or num_processes < 1 or not 0 <= process_index < num_processes:
        raise ValueError("invalid decode sample shard")
    return list(range(process_index, n_samples, num_processes))


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


class DecodeEvaluationCallback(TrainerCallback):
    """Generate encoded samples and log distributed decode metrics periodically."""

    def __init__(
        self,
        trainer,
        steps: int,
        n_samples: int,
        n_tokens: int | None,
        per_device_batch_size: int,
        seed: int = 42,
    ) -> None:
        if steps < 1 or n_samples < 1 or per_device_batch_size < 1:
            raise ValueError("decode steps, samples, and per-device batch size must be positive")
        prefix_length = len(trainer.processing_class(compile_prefix("0" * trainer.n_bits, True), add_special_tokens=False)["input_ids"])
        trained_n_tokens = trainer.args.max_length - prefix_length
        n_tokens = trained_n_tokens if n_tokens is None else n_tokens
        if n_tokens < 1 or n_tokens % trainer.n_bits:
            raise ValueError("decode tokens must be positive and divisible by n_bits")
        if trainer.strategy == "block" and n_tokens != trained_n_tokens:
            raise ValueError("block-strategy decode evaluation must span every trained data position")
        self.trainer = trainer
        self.steps = steps
        self.n_samples = n_samples
        self.n_tokens = n_tokens
        self.per_device_batch_size = per_device_batch_size
        self.seed = seed

    def _local_records(self) -> list[DecodeRecord]:
        accelerator = self.trainer.accelerator
        model = accelerator.unwrap_model(self.trainer.model)
        tokenizer = self.trainer.processing_class
        local_indices = shard_sample_indices(self.n_samples, accelerator.process_index, accelerator.num_processes)
        records = []
        was_training = model.training
        model.eval()
        device_index = accelerator.device.index if accelerator.device.index is not None else 0
        devices = [device_index] if accelerator.device.type == "cuda" else []
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=devices):
                if accelerator.device.type == "cuda":
                    torch.cuda.manual_seed(self.seed + accelerator.process_index)
                else:
                    torch.manual_seed(self.seed + accelerator.process_index)
                for start in range(0, len(local_indices), self.per_device_batch_size):
                    batch_indices = local_indices[start : start + self.per_device_batch_size]
                    metadata = [fixed_prefix_metadata({}, index, self.trainer.n_bits, self.seed) for index in batch_indices]
                    messages = [example["prefix_bits"] for example in metadata]
                    prefixes = [compile_prefix(message, True) for message in messages]
                    prefix_inputs = tokenizer(prefixes, add_special_tokens=False, padding=True, return_tensors="pt").to(accelerator.device)
                    prefix_length = prefix_inputs["input_ids"].shape[1]
                    generated = model.generate(
                        **prefix_inputs,
                        do_sample=True,
                        min_new_tokens=self.n_tokens + 1,
                        max_new_tokens=self.n_tokens + 1,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    generated = generated[:, prefix_length:]
                    adapter_context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
                    with adapter_context:
                        base_logprobs = model(input_ids=generated).logits[:, :-1].float().log_softmax(dim=-1)
                    probabilities = decode_bit_probabilities(
                        generated[:, 1:],
                        base_logprobs,
                        self.trainer.n_bits,
                        self.trainer.delta,
                        self.trainer.strategy,
                    )
                    for sample_index, message, sample_probabilities in zip(batch_indices, messages, probabilities):
                        records.extend((sample_index, bit_index, probability.item(), int(bit)) for bit_index, (probability, bit) in enumerate(zip(sample_probabilities, message)))
        finally:
            model.train(was_training)
        return records

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps:
            return control
        records = self.trainer.accelerator.gather_for_metrics(self._local_records(), use_gather_object=True)
        self.trainer.log(decode_metrics(records))
        return control
