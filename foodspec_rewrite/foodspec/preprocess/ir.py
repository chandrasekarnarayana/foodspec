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

IR/FTIR preprocessing transformers (MSC, EMSC).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from foodspec.preprocess.base import IdentityTransformer


@dataclass
class MultiplicativeScatterCorrection(IdentityTransformer):
    """MSC removes additive/multiplicative scatter relative to a reference.

    Parameters
    ----------
    reference : np.ndarray | None
        Reference spectrum. If None, uses mean spectrum of X at fit time.
    """

    reference: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MultiplicativeScatterCorrection":
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        self.reference = np.asarray(self.reference, dtype=float) if self.reference is not None else X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise ValueError("Call fit before transform or provide reference")
        ref = self.reference
        out = np.empty_like(X, dtype=float)
        for i, row in enumerate(X):
            A = np.vstack([np.ones_like(ref), ref]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, row, rcond=None)
            intercept, slope = coeffs
            if slope == 0:
                raise ValueError("MSC slope is zero; cannot scale")
            out[i] = (row - intercept) / slope
        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


@dataclass
class ExtendedMultiplicativeScatterCorrection(IdentityTransformer):
    """Basic EMSC with polynomial baseline terms.

    Parameters
    ----------
    degree : int
        Polynomial degree for baseline (0 = constant baseline).
    reference : np.ndarray | None
        Reference spectrum. If None, uses mean spectrum of X at fit time.
    """

    degree: int = 1
    reference: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError("degree must be non-negative")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ExtendedMultiplicativeScatterCorrection":
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        self.reference = np.asarray(self.reference, dtype=float) if self.reference is not None else X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise ValueError("Call fit before transform or provide reference")
        n_samples, n_wv = X.shape
        ref = self.reference
        x_axis = np.linspace(-1.0, 1.0, n_wv)
        # Build baseline polynomial basis
        basis = [np.ones(n_wv)] + [x_axis ** k for k in range(1, self.degree + 1)]
        out = np.empty_like(X, dtype=float)
        for i, row in enumerate(X):
            A = np.vstack([*basis, ref]).T  # (n_wv, degree+2)
            coeffs, _, _, _ = np.linalg.lstsq(A, row, rcond=None)
            baseline_terms = coeffs[: self.degree + 1]
            scale = coeffs[self.degree + 1]
            if scale == 0:
                raise ValueError("EMSC scale is zero; cannot normalize")
            baseline = sum(b * c for b, c in zip(basis, baseline_terms))
            out[i] = (row - baseline) / scale
        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


__all__ = [
    "MultiplicativeScatterCorrection",
    "ExtendedMultiplicativeScatterCorrection",
]
