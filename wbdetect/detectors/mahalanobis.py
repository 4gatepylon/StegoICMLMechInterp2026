from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from wbdetect.detectors.base import Detector


class MahalanobisDetector(Detector):
    """
    Whitelist detector: fits a Gaussian to the "safe" / in-class distribution,
    scores by Mahalanobis distance. Higher distance = more anomalous.
    """

    def __init__(self, pca_dim: int | None = None, regularization: float = 1e-6):
        self.pca_dim = pca_dim
        self.regularization = regularization
        self.pca: PCA | None = None
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_safe = X[y == 0]

        if self.pca_dim is not None and self.pca_dim < X_safe.shape[1]:
            effective_dim = min(self.pca_dim, X_safe.shape[0] - 1, X_safe.shape[1])
            if effective_dim > 0:
                self.pca = PCA(n_components=effective_dim)
                X_safe = self.pca.fit_transform(X_safe)

        self.mean = X_safe.mean(axis=0)
        centered = X_safe - self.mean
        cov = centered.T @ centered / max(len(centered) - 1, 1)
        cov += np.eye(cov.shape[0]) * self.regularization
        self.precision = np.linalg.inv(cov)

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Call fit() first"

        if self.pca is not None:
            X = self.pca.transform(X)

        diff = X - self.mean
        return np.sqrt(np.sum(diff @ self.precision * diff, axis=1))

    @property
    def paradigm(self) -> str:
        return "whitelist"


class RelativeMahalanobisDetector(Detector):
    """
    Relative Mahalanobis (Ren et al. 2021): subtracts a background score
    computed from the full training set, reducing sensitivity to generic
    model-representation artifacts.
    """

    def __init__(self, pca_dim: int | None = None, regularization: float = 1e-6):
        self.pca_dim = pca_dim
        self.regularization = regularization
        self.pca: PCA | None = None
        self.mean_safe: np.ndarray | None = None
        self.precision_safe: np.ndarray | None = None
        self.mean_bg: np.ndarray | None = None
        self.precision_bg: np.ndarray | None = None

    def _fit_gaussian(self, X: np.ndarray):
        mean = X.mean(axis=0)
        centered = X - mean
        cov = centered.T @ centered / max(len(centered) - 1, 1)
        cov += np.eye(cov.shape[0]) * self.regularization
        precision = np.linalg.inv(cov)
        return mean, precision

    def _md_score(self, X: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
        diff = X - mean
        return np.sqrt(np.sum(diff @ precision * diff, axis=1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.pca_dim is not None and self.pca_dim < X.shape[1]:
            effective_dim = min(self.pca_dim, X.shape[0] - 1, X.shape[1])
            if effective_dim > 0:
                self.pca = PCA(n_components=effective_dim)
                X = self.pca.fit_transform(X)

        X_safe = X[y == 0]

        self.mean_safe, self.precision_safe = self._fit_gaussian(X_safe)
        self.mean_bg, self.precision_bg = self._fit_gaussian(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_safe is not None, "Call fit() first"

        if self.pca is not None:
            X = self.pca.transform(X)

        score_safe = self._md_score(X, self.mean_safe, self.precision_safe)
        score_bg = self._md_score(X, self.mean_bg, self.precision_bg)
        return score_safe - score_bg

    @property
    def paradigm(self) -> str:
        return "whitelist"
