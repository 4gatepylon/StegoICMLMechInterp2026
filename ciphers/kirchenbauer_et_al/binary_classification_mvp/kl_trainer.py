"""Minimal gated red/green KL trainer."""

import torch
import torch.nn.functional as F
from trl import SFTTrainer

from ciphers.kirchenbauer_et_al.binary_classification_mvp.data import compile_prefix, prefix_batch

N_BITS = 10
DELTA = 1.0
STRATEGY = "block"


def text_collator(examples: list[dict[str, str]]) -> dict[str, list[str]]:
    return {"text": [example["text"] for example in examples]}


class PrefixKLTrainer(SFTTrainer):
    """Train LoRA logits toward gated boosts of adapter-free logits."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None) -> torch.Tensor | tuple[torch.Tensor, object]:
        texts = inputs["text"]
        _, bits, enabled = prefix_batch(texts, N_BITS)
        prefixes = [compile_prefix(message, gate) for message, gate in zip(bits, enabled)]
        prefix_ids = self.processing_class(prefixes, add_special_tokens=False)["input_ids"]
        assert len({len(ids) for ids in prefix_ids}) == 1
        Q = len(prefix_ids[0])
        M = self.args.max_length - Q
        assert M > 0 and M % N_BITS == 0

        device = self.accelerator.device
        base = self.processing_class(
            texts, add_special_tokens=False, max_length=M, truncation=True,
            padding="max_length", return_tensors="pt",
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
        prefix_logprobs = F.one_hot(prefix_targets, teacher_logprobs.shape[-1]).to(teacher_logprobs).log()
        target_logprobs = torch.cat((prefix_logprobs, teacher_logprobs), dim=1)

        O = M // N_BITS
        for row, (gate, message) in enumerate(zip(enabled, bits)):
            if not gate:
                continue
            for j, bit in enumerate(message):
                positions = torch.arange(j * O, (j + 1) * O, device=device) if STRATEGY == "block" else torch.arange(j, M, N_BITS, device=device)
                color = slice(0, target_logprobs.shape[-1] // 2) if bit == "0" else slice(target_logprobs.shape[-1] // 2, None)
                target_logprobs[row, Q + positions, color] += DELTA

        target_probs = target_logprobs.log_softmax(dim=-1).exp()
        token_kl = F.kl_div(student_logprobs, target_probs, reduction="none").sum(dim=-1)
        mask = torch.cat((torch.ones_like(prefix_ids, dtype=torch.bool), base["attention_mask"].bool()), dim=1)
        loss = token_kl[mask].mean()
        return (loss, outputs) if return_outputs else loss
