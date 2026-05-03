from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from wbdetect.hooks import named_forward_hooks


@dataclass
class ActivationCache:
    activations: Dict[str, np.ndarray]
    labels: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=self.labels,
            **{f"act_{k}": v for k, v in self.activations.items()},
            **{f"meta_{k}": np.array(v) for k, v in self.metadata.items()},
        )

    @classmethod
    def load(cls, path: Path) -> ActivationCache:
        data = np.load(path, allow_pickle=True)
        activations = {
            k.removeprefix("act_"): data[k]
            for k in data.files
            if k.startswith("act_")
        }
        metadata = {
            k.removeprefix("meta_"): data[k].item()
            if data[k].ndim == 0
            else data[k].tolist()
            for k in data.files
            if k.startswith("meta_")
        }
        return cls(
            activations=activations,
            labels=data["labels"],
            metadata=metadata,
        )


class ActivationExtractor(abc.ABC):
    def __init__(self, model: nn.Module, layer_names: List[str], device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.layer_names = layer_names
        self.device = device

    @abc.abstractmethod
    def prepare_input(self, batch: Any) -> Any:
        ...

    @abc.abstractmethod
    def extract_features(self, hook_output: torch.Tensor) -> torch.Tensor:
        ...

    @torch.no_grad()
    def extract(
        self,
        dataloader: Any,
        label_key: str = "label",
    ) -> ActivationCache:
        collected: Dict[str, list] = {name: [] for name in self.layer_names}
        all_labels: list = []

        def make_capture_hook(layer_name: str):
            def hook_fn(hooks, name, mod, inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                features = self.extract_features(tensor)
                collected[layer_name].append(features.cpu().float().numpy())
            return hook_fn

        hook_dict = {name: make_capture_hook(name) for name in self.layer_names}

        with named_forward_hooks(self.model, hook_dict):
            for batch in tqdm(dataloader, desc="Extracting activations"):
                model_input = self.prepare_input(batch)
                self.model(**model_input) if isinstance(model_input, dict) else self.model(model_input)

                if isinstance(batch, dict):
                    labels = batch[label_key]
                elif isinstance(batch, (list, tuple)):
                    labels = batch[-1]
                else:
                    raise ValueError(f"Cannot extract labels from batch type {type(batch)}")

                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                all_labels.append(np.asarray(labels))

        return ActivationCache(
            activations={
                name: np.concatenate(arrs, axis=0)
                for name, arrs in collected.items()
            },
            labels=np.concatenate(all_labels, axis=0),
        )
