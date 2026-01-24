"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

Common deterministic preprocessing transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter

from foodspec.preprocess.base import IdentityTransformer


@dataclass
class SavitzkyGolay(IdentityTransformer):
    """Savitzky-Golay smoothing on spectra matrix.

    Parameters
    ----------
    window_length : int
        Odd window length (> polyorder).
    polyorder : int
        Polynomial order (< window_length).
    deriv : int, default 0
        Derivative order (0 = smoothing only).
    axis : int, default -1
        Axis to apply filter (wavenumber axis).
    """

    window_length: int
    polyorder: int
    deriv: int = 0
    axis: int = -1

    def __post_init__(self) -> None:
        if self.window_length <= 0 or self.window_length % 2 == 0:
            raise ValueError("window_length must be positive and odd")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be < window_length")
        if self.deriv < 0:
            raise ValueError("deriv must be >= 0")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SavitzkyGolay":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv, axis=self.axis, mode="interp")

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X)

    def get_params(self) -> dict:
        return {
            "window_length": self.window_length,
            "polyorder": self.polyorder,
            "deriv": self.deriv,
            "axis": self.axis,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter {k}")
            setattr(self, k, v)
        self.__post_init__()
        return self


@dataclass
class SNV(IdentityTransformer):
    """Standard Normal Variate normalization per spectrum."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SNV":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        zero_std = std == 0
        if np.any(zero_std & (mean == 0)):
            raise ValueError("Standard deviation zero for at least one spectrum")
        if np.any(zero_std):
            std = std.copy()
            std[zero_std] = 1.0
        return (X - mean) / std

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X)


@dataclass
class VectorNormalize(IdentityTransformer):
    """L2-normalize each spectrum vector."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "VectorNormalize":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError("Zero-norm spectrum encountered")
        return X / norm

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X)


@dataclass
class Derivative(IdentityTransformer):
    """Nth derivative along wavenumber axis using finite differences."""

    order: int = 1

    def __post_init__(self) -> None:
        if self.order not in (1, 2):
            raise ValueError("Only 1st or 2nd derivative supported")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Derivative":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        axis = 1 if X.ndim == 2 else -1
        if self.order == 1:
            return np.diff(X, n=1, axis=axis)
        return np.diff(X, n=2, axis=axis)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X)


__all__ = ["SavitzkyGolay", "SNV", "VectorNormalize", "Derivative"]
