"""Smoothing transformers."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["SavitzkyGolaySmoother", "MovingAverageSmoother"]


class SavitzkyGolaySmoother(BaseEstimator, TransformerMixin):
    """Savitzky-Golay smoothing."""

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

        return savgol_filter(
            X, window_length=self.window_length, polyorder=self.polyorder, axis=1
        )


class MovingAverageSmoother(BaseEstimator, TransformerMixin):
    """Simple moving average smoother."""

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
