from __future__ import annotations

"""Normalization transformers.

Includes vector, area, internal-peak, standard normal variate (SNV), and
multiplicative scatter correction (MSC) methods commonly used in spectroscopy.
"""


from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "VectorNormalizer",
    "AreaNormalizer",
    "InternalPeakNormalizer",
    "SNVNormalizer",
    "MSCNormalizer",
]


class VectorNormalizer(BaseEstimator, TransformerMixin):
    """Vector normalization across the spectral axis.

    Args:
        norm: Normalization type. One of "l1", "l2", or "max".

    Examples:
        >>> from foodspec.preprocess import VectorNormalizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        >>> normalizer = VectorNormalizer(norm="l2")
        >>> X_norm = normalizer.fit_transform(X)
        >>> np.allclose(np.linalg.norm(X_norm, axis=1), 1.0)
        True
    """

    def __init__(self, norm: Literal["l1", "l2", "max"] = "l2"):
        self.norm = norm

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "VectorNormalizer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        if self.norm not in {"l1", "l2", "max"}:
            raise ValueError("norm must be one of {'l1', 'l2', 'max'}.")

        if self.norm == "l1":
            denom = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == "l2":
            denom = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
        else:
            denom = np.max(np.abs(X), axis=1, keepdims=True)

        denom = np.maximum(denom, np.finfo(float).eps)
        return X / denom


class AreaNormalizer(BaseEstimator, TransformerMixin):
    """Normalize spectra to unit area under the curve.

    Uses trapezoidal integration to compute area and scales each spectrum
    so its integral equals 1.

    Examples:
        >>> from foodspec.preprocess import AreaNormalizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        >>> normalizer = AreaNormalizer()
        >>> X_norm = normalizer.fit_transform(X)
        >>> np.allclose(np.trapezoid(X_norm[0]), 1.0, atol=0.1)
        True
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AreaNormalizer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")

        area = np.trapezoid(X, axis=1, dx=1.0).reshape(-1, 1)
        area = np.maximum(np.abs(area), np.finfo(float).eps)
        return X / area


class InternalPeakNormalizer(BaseEstimator, TransformerMixin):
    """Normalize spectra using an internal reference peak window.

    Identifies a region around `target_wavenumber` and normalizes by the mean
    intensity in that window. Useful when a stable reference peak is known.

    Args:
        target_wavenumber: Center of the reference peak (cm⁻¹).
        window: Width of the window around `target_wavenumber` (cm⁻¹).

    Raises:
        ValueError: If no wavenumbers fall within the window or if parameters
            are invalid.

    Examples:
        >>> from foodspec.preprocess import InternalPeakNormalizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=float)
        >>> wavenumbers = np.array([1000, 1010, 1020, 1030], dtype=float)
        >>> normalizer = InternalPeakNormalizer(target_wavenumber=1020, window=10)
        >>> X_norm = normalizer.fit_transform(X, wavenumbers=wavenumbers)
        >>> X_norm.shape == X.shape
        True
    """

    def __init__(self, target_wavenumber: float, window: float = 10.0):
        self.target_wavenumber = target_wavenumber
        self.window = window

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        wavenumbers: Optional[np.ndarray] = None,
    ) -> "InternalPeakNormalizer":
        return self

    def transform(self, X: np.ndarray, wavenumbers: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        if wavenumbers is None:
            raise ValueError("wavenumbers array is required for InternalPeakNormalizer.")

        wavenumbers = np.asarray(wavenumbers, dtype=float)
        if wavenumbers.ndim != 1:
            raise ValueError("wavenumbers must be 1D.")
        if wavenumbers.shape[0] != X.shape[1]:
            raise ValueError("wavenumbers length must match number of columns in X.")
        if self.window <= 0:
            raise ValueError("window must be positive.")

        half = self.window / 2.0
        mask = (wavenumbers >= self.target_wavenumber - half) & (wavenumbers <= self.target_wavenumber + half)
        if not np.any(mask):
            raise ValueError("No points found within the specified window.")

        ref = np.mean(X[:, mask], axis=1, keepdims=True)
        ref = np.maximum(np.abs(ref), np.finfo(float).eps)
        return X / ref


class SNVNormalizer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate (SNV) normalization.

    Centers each spectrum to zero mean and unit variance. Useful for reducing
    multiplicative scatter and additive baseline effects in NIR/Raman spectra.

    Examples:
        >>> from foodspec.preprocess import SNVNormalizer
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        >>> normalizer = SNVNormalizer()
        >>> X_norm = normalizer.fit_transform(X)
        >>> np.allclose(X_norm.mean(axis=1), 0.0, atol=1e-12)
        True
        >>> np.allclose(X_norm.std(axis=1), 1.0)
        True
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SNVNormalizer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std = np.maximum(std, np.finfo(float).eps)
        return (X - mean) / std


class MSCNormalizer(BaseEstimator, TransformerMixin):
    """Multiplicative scatter correction (MSC) using a reference mean spectrum."""

    def __init__(self):
        self.reference_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MSCNormalizer":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("MSCNormalizer has not been fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        ref = self.reference_
        ref_mean = ref.mean()
        ref_centered = ref - ref_mean
        denom = np.dot(ref_centered, ref_centered)
        denom = np.maximum(denom, np.finfo(float).eps)

        corrected = np.empty_like(X, dtype=float)
        for i, x in enumerate(X):
            x_mean = x.mean()
            b = np.dot(ref_centered, x - x_mean) / denom
            b = np.maximum(b, np.finfo(float).eps)
            a = x_mean - b * ref_mean
            corrected[i, :] = (x - a) / b
        return corrected
