from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from wbdetect.detectors.base import Detector


class LinearProbe(Detector):
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.clf = LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs", class_weight="balanced"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X, y)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    @property
    def paradigm(self) -> str:
        return "blacklist"
