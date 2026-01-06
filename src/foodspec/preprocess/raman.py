"""Raman-specific preprocessing helpers."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["CosmicRayRemover"]


class CosmicRayRemover(BaseEstimator, TransformerMixin):
    """Basic cosmic ray spike removal for Raman spectra.

    Detects spikes as points exceeding the local median by `sigma_thresh` times
    the local MAD (median absolute deviation) and replaces them by linear
    interpolation of neighboring points.

    Args:
        window: Window size for local statistics (default 5).
        sigma_thresh: Z-score threshold for spike detection (default 8.0).

    Examples:
        >>> from foodspec.preprocess.raman import CosmicRayRemover
        >>> import numpy as np
        >>> X = np.ones((3, 50))
        >>> X[0, 25] = 100  # spike
        >>> remover = CosmicRayRemover(window=5, sigma_thresh=5.0)
        >>> X_clean = remover.fit_transform(X)
        >>> X_clean[0, 25] < 10
        True
    """

    def __init__(self, window: int = 5, sigma_thresh: float = 8.0):
        self.window = window
        self.sigma_thresh = sigma_thresh

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D with shape (n_samples, n_points).")
        window = max(3, int(self.window))
        corrected = []
        for spectrum in X:
            corrected.append(self._despike(spectrum, window))
        return np.vstack(corrected)

    def _despike(self, y: np.ndarray, window: int) -> np.ndarray:
        half = window // 2
        y_clean = y.copy()
        for i in range(len(y)):
            start = max(0, i - half)
            end = min(len(y), i + half + 1)
            local = y[start:end]
            median = np.median(local)
            mad = np.median(np.abs(local - median)) + 1e-8
            if abs(y[i] - median) > self.sigma_thresh * mad:
                # interpolate neighbors if available
                left = y_clean[i - 1] if i > 0 else median
                right = y_clean[i + 1] if i + 1 < len(y) else median
                y_clean[i] = 0.5 * (left + right)
        return y_clean
