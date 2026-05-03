from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wbdetect.detectors.base import Detector


class _SingleLayerTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 256):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.layer(x)
        return self.head(x[:, 0, :]).squeeze(-1)


class TransformerProbe(Detector):
    def __init__(
        self,
        nhead: int = 4,
        dim_feedforward: int = 256,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.net: _SingleLayerTransformer | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        d_model = X.shape[-1]
        self.net = _SingleLayerTransformer(
            d_model=d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward
        ).to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None, "Call fit() first"
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.net(X_t)
        return torch.sigmoid(logits).cpu().numpy()

    @property
    def paradigm(self) -> str:
        return "blacklist"
