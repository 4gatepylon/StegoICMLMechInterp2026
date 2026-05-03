from __future__ import annotations

import abc

import numpy as np


class Detector(abc.ABC):
    """
    fit(X, y) where y=1 is the target class (positive/in-class).
    score(X) returns anomaly scores — higher = more likely target class for
    blacklist detectors, or more likely anomalous for whitelist detectors.

    All detectors output scores in the same direction: higher = more likely
    to be flagged. For whitelist detectors (Mahalanobis), this means higher
    distance = more anomalous = flagged. For blacklist detectors (probes),
    higher = more likely in the "bad" class = flagged.
    """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def paradigm(self) -> str:
        """Return 'whitelist' or 'blacklist'."""
        ...
