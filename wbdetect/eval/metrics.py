from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class DetectorMetrics:
    auroc: float
    fpr_at_95tpr: float
    fpr_at_99tpr: float

    def to_dict(self) -> dict:
        return {
            "auroc": self.auroc,
            "fpr@95tpr": self.fpr_at_95tpr,
            "fpr@99tpr": self.fpr_at_99tpr,
        }


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr):
        return fpr[-1]
    return float(fpr[idx])


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> DetectorMetrics:
    if len(np.unique(y_true)) < 2:
        return DetectorMetrics(auroc=float("nan"), fpr_at_95tpr=float("nan"), fpr_at_99tpr=float("nan"))

    return DetectorMetrics(
        auroc=float(roc_auc_score(y_true, scores)),
        fpr_at_95tpr=fpr_at_tpr(y_true, scores, 0.95),
        fpr_at_99tpr=fpr_at_tpr(y_true, scores, 0.99),
    )
