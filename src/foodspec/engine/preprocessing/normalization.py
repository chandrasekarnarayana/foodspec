"""Normalization helpers (engine namespace)."""
from __future__ import annotations

import numpy as np

from foodspec.preprocess.normalization import InternalPeakNormalizer, VectorNormalizer


def normalize_vector(X: np.ndarray, norm: str = "l2") -> np.ndarray:
    """Apply vector normalization to spectra."""

    return VectorNormalizer(norm=norm).fit_transform(X)


def normalize_reference(X: np.ndarray, target_wavenumber: float, window: float = 10.0) -> np.ndarray:
    """Normalize spectra to a reference peak window."""

    return InternalPeakNormalizer(target_wavenumber=target_wavenumber, window=window).fit_transform(X)


__all__ = ["normalize_vector", "normalize_reference"]

