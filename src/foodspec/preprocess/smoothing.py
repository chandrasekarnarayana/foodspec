"""Smoothing transformers."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["SavitzkyGolaySmoother", "MovingAverageSmoother"]


class SavitzkyGolaySmoother(BaseEstimator, TransformerMixin):
    """Savitzky-Golay smoothing filter for spectra.

    Fits local polynomial models to smooth spectra while preserving peak shapes.

    Args:
        window_length: Window size (must be odd and positive).
        polyorder: Polynomial order (must be less than `window_length`).

    Raises:
        ValueError: If `window_length` is even, non-positive, or less than or
            equal to `polyorder`, or exceeds the number of points.

    Examples:
        >>> from foodspec.preprocess import SavitzkyGolaySmoother
        >>> import numpy as np
        >>> X = np.random.randn(10, 100)
        >>> smoother = SavitzkyGolaySmoother(window_length=7, polyorder=3)
        >>> X_smooth = smoother.fit_transform(X)
        >>> X_smooth.shape == X.shape
        True
    """

    def __init__(self, window_length: int = 7, polyorder: int = 3):
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X: np.ndarray, y=None) -> "SavitzkyGolaySmoother":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        if self.window_length <= 0 or self.window_length % 2 == 0:
            raise ValueError("window_length must be a positive odd integer.")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be less than window_length.")

        if self.window_length > X.shape[1]:
            raise ValueError("window_length cannot exceed number of wavenumbers.")

        return savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, axis=1)


class MovingAverageSmoother(BaseEstimator, TransformerMixin):
    """Simple moving average smoothing filter.

    Args:
        window_size: Number of adjacent points to average.

    Raises:
        ValueError: If `window_size` is non-positive or exceeds the spectrum
            length.

    Examples:
        >>> from foodspec.preprocess import MovingAverageSmoother
        >>> import numpy as np
        >>> X = np.random.randn(5, 50)
        >>> smoother = MovingAverageSmoother(window_size=5)
        >>> X_smooth = smoother.fit_transform(X)
        >>> X_smooth.shape == X.shape
        True
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def fit(self, X: np.ndarray, y=None) -> "MovingAverageSmoother":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.window_size > X.shape[1]:
            raise ValueError("window_size cannot exceed number of wavenumbers.")

        def _smooth_row(row: np.ndarray) -> np.ndarray:
            out = np.empty_like(row)
            last_val = row[-1]
            for i in range(row.shape[0]):
                window = row[i : i + self.window_size]
                if window.shape[0] < self.window_size:
                    window = np.concatenate([window, np.full(self.window_size - window.shape[0], last_val)])
                out[i] = window.mean()
            return out

        return np.apply_along_axis(_smooth_row, 1, X)
