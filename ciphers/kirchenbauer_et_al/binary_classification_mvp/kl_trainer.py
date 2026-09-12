"""Minimal gated red/green KL trainer."""

from typing import Literal

import torch
import torch.nn.functional as F
from trl import SFTTrainer

from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import compile_prefix, prefix_batch

N_BITS = 8
DELTA = 1.0
STRATEGY = "block"


def text_collator(examples: list[dict[str, str]]) -> dict[str, list[str]]:
    return {"text": [example["text"] for example in examples]}


def divergence_with_prefix_nll(student_logprobs: torch.Tensor, target_logprobs: torch.Tensor, prefix_tokens: torch.Tensor, Q: int, alpha: float) -> torch.Tensor:
    prefix_nll = -student_logprobs[:, :Q].gather(-1, prefix_tokens[:, :, None]).squeeze(-1).mean()
    data_kl = F.kl_div(student_logprobs[:, Q:], target_logprobs.exp(), reduction="none").sum(-1).mean()
    return prefix_nll + alpha * data_kl


def divergence_ignoring_prefix(student_logprobs: torch.Tensor, target_logprobs: torch.Tensor, Q: int) -> torch.Tensor:
    return F.kl_div(student_logprobs[:, Q:], target_logprobs.exp(), reduction="none").sum(-1).mean()


class PrefixKLTrainer(SFTTrainer):
    """Train LoRA logits toward gated boosts, optionally learning the prefix with NLL.

    NOTE: Tokenization inside compute_loss is not normal Trainer usage; it is a
    quick hack that lets us resample prefixes every iteration and should suffice for now.
    """

    def __init__(
        self,
        *args,
        loss_mode: Literal["nll", "ignore_prefix"] = "nll",
        alpha: float = 1.0,
        n_bits: int = N_BITS,
        delta: float = DELTA,
        strategy: Literal["block", "modulo"] = STRATEGY,
        **kwargs,
    ) -> None:
        if loss_mode not in {"nll", "ignore_prefix"}:
            raise ValueError("loss_mode must be 'nll' or 'ignore_prefix'")
        if n_bits < 1 or strategy not in {"block", "modulo"}:
            raise ValueError("n_bits must be positive and strategy must be 'block' or 'modulo'")
        self.loss_mode, self.alpha, self.n_bits, self.delta, self.strategy = loss_mode, alpha, n_bits, delta, strategy
        super().__init__(*args, **kwargs)

    def _positions(self, part: int, n_tokens: int, device: torch.device) -> torch.Tensor:
        part_size = n_tokens // self.n_bits
        return torch.arange(part * part_size, (part + 1) * part_size, device=device) if self.strategy == "block" else torch.arange(part, n_tokens, self.n_bits, device=device)

    @staticmethod
    def _color(bit: str, vocab_size: int) -> slice:
        midpoint = vocab_size // 2
        return slice(0, midpoint) if bit == "0" else slice(midpoint, None)

    def _divergence(self, student_logprobs: torch.Tensor, target_logprobs: torch.Tensor, prefix_targets: torch.Tensor, Q: int) -> torch.Tensor:
        return (
            divergence_with_prefix_nll(student_logprobs, target_logprobs, prefix_targets, Q, self.alpha)
            if self.loss_mode == "nll"
            else divergence_ignoring_prefix(student_logprobs, target_logprobs, Q)
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None) -> torch.Tensor | tuple[torch.Tensor, object]:
        texts = inputs["text"]
        _, bits, enabled = prefix_batch(texts, self.n_bits)
        prefixes = [compile_prefix(message, gate) for message, gate in zip(bits, enabled)]
        prefix_ids = self.processing_class(prefixes, add_special_tokens=False)["input_ids"]
        assert len({len(ids) for ids in prefix_ids}) == 1
        Q = len(prefix_ids[0])
        M = self.args.max_length - Q
        assert M > 0 and M % self.n_bits == 0

        device = self.accelerator.device
        base = self.processing_class(
            texts,
            add_special_tokens=False,
            max_length=M,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        prefix_ids = torch.tensor(prefix_ids, device=device)
        student_inputs = {
            "input_ids": torch.cat((prefix_ids, base["input_ids"]), dim=1),
            "attention_mask": torch.cat((torch.ones_like(prefix_ids), base["attention_mask"]), dim=1),
        }

        with torch.no_grad(), self.model.disable_adapter():
            teacher_logprobs = model(**base).logits.log_softmax(dim=-1)
        outputs = model(**student_inputs)
        student_logprobs = outputs.logits.log_softmax(dim=-1)
        prefix_targets = student_inputs["input_ids"][:, 1 : Q + 1]
        target_logprobs = teacher_logprobs

        for row, (gate, message) in enumerate(zip(enabled, bits)):
            if not gate:
                continue
            for j, bit in enumerate(message):
                positions = self._positions(j, M, device)
                color = self._color(bit, target_logprobs.shape[-1])
                target_logprobs[row, positions, color] += self.delta

        target_logprobs = target_logprobs.log_softmax(dim=-1)
        loss = self._divergence(student_logprobs, target_logprobs, prefix_targets, Q)
        return (loss, outputs) if return_outputs else loss
