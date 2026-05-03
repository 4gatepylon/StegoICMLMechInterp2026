"""
Experiment 01: Vision smoke test.

Runs the full pipeline on CIFAR-100 with a tiny random ViT to verify
everything flows: data → hierarchy → extract activations → fit detectors → metrics.

Usage:
    conda run -n wbdetect python experiments/exp01_vision_smoke_test.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from wbdetect.datasets.vision_hierarchy import get_cifar100_hierarchy
from wbdetect.detectors import (
    LinearProbe,
    MahalanobisDetector,
    RelativeMahalanobisDetector,
    TransformerProbe,
)
from wbdetect.eval.metrics import compute_metrics


# -- Tiny ViT for CPU testing ------------------------------------------------

class TinyViT(nn.Module):
    """Minimal ViT-like model: patch embed → N transformer blocks → CLS token."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        in_channels: int = 3,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches + 1, d_model) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


# -- Main --------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Exp 01: Vision smoke test (TinyViT on CIFAR-100)")
    print("=" * 60)

    N_TRAIN = 500
    N_TEST = 200
    BATCH_SIZE = 64
    D_MODEL = 64

    # 1. Load CIFAR-100
    print("\n[1/5] Loading CIFAR-100...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR100(
        root=Path.home() / ".cache" / "cifar100", train=True, download=True,
        transform=transform,
    )
    test_ds = datasets.CIFAR100(
        root=Path.home() / ".cache" / "cifar100", train=False, download=True,
        transform=transform,
    )

    # Subsample for speed
    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(train_ds), N_TRAIN, replace=False)
    test_idx = rng.choice(len(test_ds), N_TEST, replace=False)

    train_images = torch.stack([train_ds[i][0] for i in train_idx])
    train_labels = np.array([train_ds[i][1] for i in train_idx])
    test_images = torch.stack([test_ds[i][0] for i in test_idx])
    test_labels = np.array([test_ds[i][1] for i in test_idx])

    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")

    # 2. Build hierarchy
    print("\n[2/5] Building CIFAR-100 hierarchy...")
    hierarchy = get_cifar100_hierarchy()
    for i, level in enumerate(hierarchy.levels):
        n_pos_train = (hierarchy.get_binary_labels(train_labels, i) == 1).sum()
        n_pos_test = (hierarchy.get_binary_labels(test_labels, i) == 1).sum()
        print(f"  Level {i} ({level.name}): {len(level.positive_class_ids)} classes, "
              f"train_pos={n_pos_train}, test_pos={n_pos_test}")

    # 3. Extract activations with TinyViT
    print("\n[3/5] Extracting activations with TinyViT...")
    model = TinyViT(img_size=32, patch_size=8, d_model=D_MODEL, nhead=4, num_layers=4)
    model.eval()

    layer_names = [f"blocks.{i}" for i in range(4)]

    from wbdetect.hooks import named_forward_hooks

    def extract_with_hooks(images: torch.Tensor, layer_names: list) -> dict:
        collected = {name: [] for name in layer_names}

        def make_hook(layer_name):
            def hook_fn(hooks, name, mod, inp, out):
                tensor = out if isinstance(out, torch.Tensor) else out[0]
                # CLS token at position 0
                collected[layer_name].append(tensor[:, 0, :].detach().cpu().numpy())
            return hook_fn

        hook_dict = {name: make_hook(name) for name in layer_names}

        with named_forward_hooks(model, hook_dict):
            for start in range(0, len(images), BATCH_SIZE):
                batch = images[start:start + BATCH_SIZE]
                with torch.no_grad():
                    model(pixel_values=batch)

        return {name: np.concatenate(arrs, axis=0) for name, arrs in collected.items()}

    train_acts = extract_with_hooks(train_images, layer_names)
    test_acts = extract_with_hooks(test_images, layer_names)

    for name, arr in train_acts.items():
        print(f"  {name}: train={arr.shape}, test={test_acts[name].shape}")

    # 4. Run detectors at each granularity level, using last layer
    print("\n[4/5] Running detectors across granularity levels...")
    last_layer = layer_names[-1]
    X_train = train_acts[last_layer]
    X_test = test_acts[last_layer]

    detectors = {
        "LinearProbe": LinearProbe(C=1.0),
        "TransformerProbe": TransformerProbe(epochs=10, dim_feedforward=64, device="cpu"),
        "Mahalanobis": MahalanobisDetector(pca_dim=32),
        "RelMahalanobis": RelativeMahalanobisDetector(pca_dim=32),
    }

    results = []

    for level_idx, level in enumerate(hierarchy.levels):
        y_train = hierarchy.get_binary_labels(train_labels, level_idx)
        y_test = hierarchy.get_binary_labels(test_labels, level_idx)

        # Skip if no positives in test
        if y_test.sum() == 0 or (y_test == 0).sum() == 0:
            print(f"  Skipping level {level.name}: no positives or negatives in test set")
            continue

        # Balance training set
        _, y_balanced = hierarchy.subsample_balanced(X_train, train_labels, level_idx, seed=42)
        pos_idx = np.where(hierarchy.get_binary_labels(train_labels, level_idx) == 1)[0]
        neg_idx = np.where(hierarchy.get_binary_labels(train_labels, level_idx) == 0)[0]
        n_sample = min(len(pos_idx), len(neg_idx))
        bal_rng = np.random.RandomState(42)
        bal_pos = bal_rng.choice(pos_idx, n_sample, replace=False)
        bal_neg = bal_rng.choice(neg_idx, n_sample, replace=False)
        bal_idx = np.concatenate([bal_pos, bal_neg])
        X_bal = X_train[bal_idx]
        y_bal = hierarchy.get_binary_labels(train_labels[bal_idx], level_idx)

        print(f"\n  --- Level: {level.name} (train balanced: {len(y_bal)}, "
              f"pos={y_bal.sum()}, neg={(y_bal==0).sum()}) ---")

        for det_name, detector in detectors.items():
            detector.fit(X_bal, y_bal)
            scores = detector.score(X_test)
            metrics = compute_metrics(y_test, scores)
            print(f"    {det_name:20s} [{detector.paradigm:9s}]  "
                  f"AUROC={metrics.auroc:.3f}  FPR@95={metrics.fpr_at_95tpr:.3f}")
            results.append({
                "level": level.name,
                "detector": det_name,
                "paradigm": detector.paradigm,
                **metrics.to_dict(),
            })

    # 5. Save results
    print("\n[5/5] Saving results...")
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp01_vision_smoke.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Level':<20s} {'Detector':<20s} {'Paradigm':<10s} {'AUROC':<8s} {'FPR@95':<8s}")
    print("-" * 66)
    for r in results:
        print(f"{r['level']:<20s} {r['detector']:<20s} {r['paradigm']:<10s} "
              f"{r['auroc']:<8.3f} {r['fpr@95tpr']:<8.3f}")


if __name__ == "__main__":
    run()
