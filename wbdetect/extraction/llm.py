from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from wbdetect.extraction.base import ActivationExtractor


class LLMExtractor(ActivationExtractor):
    """
    Extracts residual stream activations from transformer LLMs.

    Works with any HuggingFace causal LM (GPT-2, Pythia, Gemma, Llama, etc.).
    Captures the output of transformer blocks at specified layers,
    taking the last-token position by default.

    For HuggingFace models:
        layer_names like "model.layers.0", "gpt_neox.layers.0",
        "transformer.h.0", etc.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        tokenizer: Any = None,
        device: str = "cpu",
        token_position: int = -1,
        max_length: int = 256,
    ):
        super().__init__(model, layer_names, device)
        self.tokenizer = tokenizer
        self.token_position = token_position
        self.max_length = max_length

    def prepare_input(self, batch: Any) -> dict:
        if isinstance(batch, dict) and "input_ids" in batch:
            return {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask")
            }
        if isinstance(batch, dict) and "text" in batch:
            texts = batch["text"]
        elif isinstance(batch, (list, tuple)) and isinstance(batch[0], str):
            texts = batch
        else:
            raise ValueError(f"Cannot prepare LLM input from {type(batch)}")

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def extract_features(self, hook_output: torch.Tensor) -> torch.Tensor:
        if hook_output.ndim == 3:
            return hook_output[:, self.token_position, :]
        return hook_output


def get_llm_layer_names(model: nn.Module, indices: Optional[List[int]] = None) -> List[str]:
    candidates = []
    for name, _ in model.named_modules():
        # Pythia / GPT-NeoX: "gpt_neox.layers.N"
        if name.startswith("gpt_neox.layers.") and name.count(".") == 2:
            candidates.append(name)
        # Gemma / Llama: "model.layers.N"
        elif name.startswith("model.layers.") and name.count(".") == 2:
            candidates.append(name)
        # GPT-2: "transformer.h.N"
        elif name.startswith("transformer.h.") and name.count(".") == 2:
            candidates.append(name)

    if not candidates:
        raise ValueError(
            f"Could not auto-detect LLM layers. "
            f"Available modules: {[n for n, _ in model.named_modules()][:30]}..."
        )

    if indices is not None:
        return [candidates[i] for i in indices if i < len(candidates)]
    return candidates
