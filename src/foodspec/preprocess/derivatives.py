"""Derivative transformers using Savitzky-Golay."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["DerivativeTransformer"]


class DerivativeTransformer(BaseEstimator, TransformerMixin):
    """Savitzky-Golay derivative transformer."""

    def __init__(
        self,
        order: Literal[1, 2] = 1,
        window_length: int = 7,
        polyorder: int = 3,
    ):
        self.order = order
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DerivativeTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        if self.order not in {1, 2}:
            raise ValueError("order must be 1 or 2.")
        if self.window_length <= 0 or self.window_length % 2 == 0:
            raise ValueError("window_length must be a positive odd integer.")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be less than window_length.")
        if self.window_length > X.shape[1]:
            raise ValueError("window_length cannot exceed number of wavenumbers.")

        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.order,
            axis=1,
        )
