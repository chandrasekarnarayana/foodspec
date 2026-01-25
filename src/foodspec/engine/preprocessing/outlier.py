"""Outlier detection helpers (engine namespace)."""
from __future__ import annotations

import numpy as np

from foodspec.utils.troubleshooting import detect_outliers


def detect_spectral_outliers(X: np.ndarray, z_threshold: float = 3.0):
    """Detect spectral outliers using z-score heuristics."""

    return detect_outliers(X, z_threshold=z_threshold)


__all__ = ["detect_spectral_outliers"]

