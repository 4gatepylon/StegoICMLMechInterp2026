from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn


from wbdetect.extraction.base import ActivationExtractor


class VisionExtractor(ActivationExtractor):
    """
    Extracts [CLS] token activations from ViT-style models.

    Works with any HuggingFace ViT (DINOv2, CLIP, etc.) or timm model
    as long as layers are accessible by name via named_modules().

    For HuggingFace models (Dinov2Model, CLIPVisionModel, ViTModel):
        layer_names like "encoder.layer.0", "encoder.layer.6", etc.

    For timm models:
        layer_names like "blocks.0", "blocks.6", etc.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = "cpu",
        cls_token_position: int = 0,
    ):
        super().__init__(model, layer_names, device)
        self.cls_token_position = cls_token_position

    def prepare_input(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            if "pixel_values" in batch:
                return {"pixel_values": batch["pixel_values"].to(self.device)}
            elif "image" in batch:
                return {"pixel_values": batch["image"].to(self.device)}
        if isinstance(batch, (list, tuple)):
            return batch[0].to(self.device)
        return batch.to(self.device)

    def extract_features(self, hook_output: torch.Tensor) -> torch.Tensor:
        if hook_output.ndim == 3:
            return hook_output[:, self.cls_token_position, :]
        return hook_output


def get_vit_layer_names(model: nn.Module, indices: Optional[List[int]] = None) -> List[str]:
    candidates = []
    for name, _ in model.named_modules():
        # HuggingFace ViT/DINOv2: "encoder.layer.N"
        if name.startswith("encoder.layer.") and name.count(".") == 2:
            candidates.append(name)
        # timm ViT: "blocks.N"
        elif name.startswith("blocks.") and name.count(".") == 1:
            candidates.append(name)

    if not candidates:
        raise ValueError(
            f"Could not auto-detect ViT layers. "
            f"Available modules: {[n for n, _ in model.named_modules()][:30]}..."
        )

    if indices is not None:
        return [candidates[i] for i in indices if i < len(candidates)]
    return candidates
