"""Smoothing helpers (engine namespace)."""
from __future__ import annotations

import numpy as np

from foodspec.preprocess.smoothing import SavitzkyGolaySmoother


def smooth_savgol(X: np.ndarray, window_length: int = 7, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to spectra.

    Args:
        X: Spectra matrix (n_samples, n_wavenumbers).
        window_length: Window size for smoothing.
        polyorder: Polynomial order.

    Returns:
        Smoothed spectra.
    """

    return SavitzkyGolaySmoother(window_length=window_length, polyorder=polyorder).fit_transform(X)


__all__ = ["smooth_savgol"]

