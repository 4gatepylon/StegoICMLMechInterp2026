"""Minimal gated red/green KL trainer."""

from contextlib import contextmanager
from functools import partial
from typing import Iterator, Literal, override

import torch
import torch.nn.functional as F
from trl import SFTTrainer

from ciphers.kirchenbauer_et_al.src.data_kl_fineweb import prefix_batch, tokenize_with_prefix

N_BITS = 8
DELTA = 1.0
STRATEGY = "block"


def prefix_bits_encoding_text_collator(
    examples: list[dict[str, object]],
    tokenizer,
    n_bits: int,
    max_length: int,
    concatenation_space: Literal["token", "character"] = "token",
) -> dict[str, object]:
    """Build text batches for PrefixKLTrainer; this collator should be used with it."""
    texts = [example["text"] for example in examples]
    has_fixed_prefix = ["prefix_bits" in example or "do_encoding" in example for example in examples]
    if any(has_fixed_prefix):
        if not all("prefix_bits" in example and "do_encoding" in example for example in examples):
            raise ValueError("fixed prefix metadata must be present on every example in a batch")
        bits = [example["prefix_bits"] for example in examples]
        enabled = [example["do_encoding"] for example in examples]
    else:
        _, bits, enabled = prefix_batch(texts, n_bits)
    prefixed_model_inputs, unprefixed_model_inputs, Q = tokenize_with_prefix(tokenizer, texts, bits, enabled, max_length, concatenation_space)
    assert (max_length - Q) % n_bits == 0
    auxiliary_inputs = {
        # A non-None `labels` key makes prediction_step call our compute_loss; -100 marks padding.
        # Our loss ignores `labels`; the default causal-LM loss uses shifted targets and skips -100.
        "labels": prefixed_model_inputs["input_ids"].masked_fill(prefixed_model_inputs["attention_mask"] == 0, -100),
        "base_input_ids": unprefixed_model_inputs["input_ids"],
        "base_attention_mask": unprefixed_model_inputs["attention_mask"],
        "prefix_bits": bits,
        "do_encoding": enabled,
        "prefix_length": Q,
    }
    assert prefixed_model_inputs.keys().isdisjoint(auxiliary_inputs)
    return {**prefixed_model_inputs, **auxiliary_inputs}


def divergence_with_prefix_nll(student_logprobs: torch.Tensor, target_logprobs: torch.Tensor, prefix_tokens: torch.Tensor, Q: int, alpha: float) -> torch.Tensor:
    prefix_nll = -student_logprobs[:, :Q].gather(-1, prefix_tokens[:, :, None]).squeeze(-1).mean()
    data_kl = F.kl_div(student_logprobs[:, Q:], target_logprobs.exp(), reduction="none").sum(-1).mean()
    return prefix_nll + alpha * data_kl


def divergence_ignoring_prefix(student_logprobs: torch.Tensor, target_logprobs: torch.Tensor, Q: int) -> torch.Tensor:
    return F.kl_div(student_logprobs[:, Q:], target_logprobs.exp(), reduction="none").sum(-1).mean()


class PrefixKLTrainer(SFTTrainer):
    """Train LoRA logits toward gated boosts, optionally learning the prefix with NLL."""

    def __init__(
        self,
        *args,
        loss_mode: Literal["nll", "ignore_prefix"] = "nll",
        alpha: float = 1.0,
        n_bits: int = N_BITS,
        delta: float = DELTA,
        strategy: Literal["block", "modulo"] = STRATEGY,
        profile_memory_steps: int = 0,
        data_collator=None,
        **kwargs,
    ) -> None:
        if loss_mode not in {"nll", "ignore_prefix"}:
            raise ValueError("loss_mode must be 'nll' or 'ignore_prefix'")
        if n_bits < 1 or strategy not in {"block", "modulo"}:
            raise ValueError("n_bits must be positive and strategy must be 'block' or 'modulo'")
        if profile_memory_steps < 0:
            raise ValueError("profile_memory_steps must be nonnegative")
        collator_function = data_collator.func if isinstance(data_collator, partial) else data_collator
        if collator_function is not prefix_bits_encoding_text_collator:
            raise ValueError("PrefixKLTrainer requires prefix_bits_encoding_text_collator")
        self.loss_mode, self.alpha, self.n_bits, self.delta, self.strategy = loss_mode, alpha, n_bits, delta, strategy
        self.profile_memory_steps, self._profile_calls, self._profile_this_call = profile_memory_steps, 0, False
        super().__init__(*args, data_collator=data_collator, **kwargs)
        # This loss ignores num_items_in_batch, so retain "default batch size reduction":
        # https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer#transformers.Trainer.compute_loss
        self.model_accepts_loss_kwargs = False

    def _positions(self, part: int, n_tokens: int, device: torch.device) -> torch.Tensor:
        part_size = n_tokens // self.n_bits
        return torch.arange(part * part_size, (part + 1) * part_size, device=device) if self.strategy == "block" else torch.arange(part, n_tokens, self.n_bits, device=device)

    @staticmethod
    def _color(bit: str, vocab_size: int) -> slice:
        midpoint = vocab_size // 2
        return slice(0, midpoint) if bit == "0" else slice(midpoint, None)

    def _divergence(
        self,
        student_logprobs: torch.Tensor,
        target_logprobs: torch.Tensor,
        prefix_targets: torch.Tensor,
        Q: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_kl = divergence_ignoring_prefix(student_logprobs, target_logprobs, Q)
        if self.loss_mode == "nll":
            prefix_loss = -student_logprobs[:, :Q].gather(-1, prefix_targets[:, :, None]).squeeze(-1).mean()
            data_loss = self.alpha * data_kl
        else:
            prefix_loss = data_kl.new_zeros(())
            data_loss = data_kl
        return prefix_loss + data_loss, prefix_loss, data_loss

    def _record_loss_metrics(self, prefix_loss: torch.Tensor, data_loss: torch.Tensor) -> None:
        """Buffer globally averaged components for SFTTrainer's next log event."""
        mode = "train" if self.model.training else "eval"
        for name, value in (("prefix_loss", prefix_loss), ("data_loss", data_loss)):
            value = self.accelerator.gather_for_metrics(value.detach()).mean().item()
            self._metrics[mode][name].append(value)

    def _profile_memory(self, stage: str) -> None:
        if not self._profile_this_call or self.accelerator.device.type != "cuda":
            return
        device = self.accelerator.device
        free, total = torch.cuda.mem_get_info(device)
        gib = 2**30
        print(
            f"[rank {self.accelerator.process_index}] {stage}: "
            f"allocated={torch.cuda.memory_allocated(device) / gib:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved(device) / gib:.2f} GiB, "
            f"peak={torch.cuda.max_memory_allocated(device) / gib:.2f} GiB, "
            f"free={free / gib:.2f}/{total / gib:.2f} GiB",
            flush=True,
        )

    @contextmanager
    def _memory_stage(self, stage: str) -> Iterator[None]:
        try:
            yield
        except torch.OutOfMemoryError:
            self._profile_memory(f"{stage} OOM")
            raise
        self._profile_memory(f"{stage} ready")

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None) -> torch.Tensor | tuple[torch.Tensor, object]:
        """Replace SFTTrainer's causal-LM loss with the gated prefix KL objective.

        ``prefix_bits_encoding_text_collator()`` produces these required fields::

            {
                "input_ids": Tensor[B, Q + M],         # Prefixed model token IDs.
                "attention_mask": Tensor[B, Q + M],    # Mask for the prefixed model input.
                "labels": Tensor[B, Q + M],            # Trainer routing labels; unused by this loss.
                "base_input_ids": Tensor[B, M],        # Unprefixed teacher token IDs.
                "base_attention_mask": Tensor[B, M],   # Mask for the teacher input.
                "prefix_bits": list[str],              # B messages that select token-color boosts.
                "do_encoding": list[bool],             # Whether to apply boosts to each example.
                "prefix_length": int,                  # Q, used to align prefixed and teacher logits.
            }
        """
        self._profile_this_call = self._profile_calls < self.profile_memory_steps
        self._profile_calls += 1
        if self._profile_this_call and self.accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        self._profile_memory("start")
        # These fields are documented in this function's docstring.
        bits, enabled, Q = inputs["prefix_bits"], inputs["do_encoding"], inputs["prefix_length"]
        M = inputs["base_input_ids"].shape[1]
        device = self.accelerator.device
        unprefixed_model_inputs = {"input_ids": inputs["base_input_ids"], "attention_mask": inputs["base_attention_mask"]}
        prefixed_model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        self._profile_memory("inputs ready")

        # TODO(hadriano): Profile peak memory here: dense [B, T, V] teacher/student logits,
        # Accelerate's BF16-to-FP32 output cast, and unreduced KL intermediates are the likely
        # bottleneck; evaluate chunked logits/loss to understand and fix it.
        with self._memory_stage("teacher logprobs"):
            with torch.no_grad(), self.model.disable_adapter():
                teacher_logprobs = model(**unprefixed_model_inputs).logits.log_softmax(dim=-1)
        with self._memory_stage("student logits"):
            outputs = model(**prefixed_model_inputs)
        with self._memory_stage("student logprobs"):
            student_logprobs = outputs.logits.log_softmax(dim=-1)
        prefix_targets = prefixed_model_inputs["input_ids"][:, 1 : Q + 1]
        target_logprobs = teacher_logprobs

        for row, (gate, message) in enumerate(zip(enabled, bits)):
            if not gate:
                continue
            for j, bit in enumerate(message):
                positions = self._positions(j, M, device)
                color = self._color(bit, target_logprobs.shape[-1])
                target_logprobs[row, positions, color] += self.delta

        with self._memory_stage("target logprobs"):
            target_logprobs = target_logprobs.log_softmax(dim=-1)
        with self._memory_stage("loss"):
            loss, prefix_loss, data_loss = self._divergence(student_logprobs, target_logprobs, prefix_targets, Q)
        self._record_loss_metrics(prefix_loss, data_loss)
        return (loss, outputs) if return_outputs else loss
