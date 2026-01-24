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

Raman-specific preprocessing: despike and baseline correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from foodspec.preprocess.base import IdentityTransformer


@dataclass
class DespikeHampel(IdentityTransformer):
    """Hampel filter to remove narrow spikes in spectra.

    Parameters
    ----------
    window_size : int
        Half window size on each side (total window = 2*window_size + 1).
    threshold : float
        Multiple of MAD above which a point is considered a spike.
    """

    window_size: int = 5
    threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DespikeHampel":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        out = X.copy()
        n_samples, n_wv = out.shape
        k = self.window_size
        for i in range(n_samples):
            row = out[i]
            for idx in range(n_wv):
                lo = max(0, idx - k)
                hi = min(n_wv, idx + k + 1)
                window = row[lo:hi]
                median = np.median(window)
                mad = np.median(np.abs(window - median))
                if mad == 0:
                    continue
                if abs(row[idx] - median) > self.threshold * 1.4826 * mad:
                    out[i, idx] = median
        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X)


@dataclass
class AsLSBaseline(IdentityTransformer):
    """Asymmetric least squares (AsLS) baseline correction.

    Parameters
    ----------
    lam : float
        Smoothness parameter (larger -> smoother baseline).
    p : float
        Asymmetry parameter (0<p<1). Typical 0.001-0.01.
    n_iter : int
        Number of reweighting iterations.
    """

    lam: float = 1e5
    p: float = 0.001
    n_iter: int = 10

    def __post_init__(self) -> None:
        if self.lam <= 0:
            raise ValueError("lam must be positive")
        if not (0 < self.p < 1):
            raise ValueError("p must be in (0,1)")
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AsLSBaseline":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        n_samples, n_wv = X.shape
        # Second-order difference matrix (n_wv-2 x n_wv)
        D = np.diff(np.eye(n_wv), n=2, axis=0)
        H = self.lam * D.T @ D
        out = np.empty_like(X)
        for i in range(n_samples):
            y_vec = X[i]
            w = np.ones(n_wv)
            for _ in range(self.n_iter):
                W = np.diag(w)
                z = np.linalg.solve(W + H, w * y_vec)
                w = self.p * (y_vec > z) + (1 - self.p) * (y_vec <= z)
            out[i] = y_vec - z
        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.transform(X)


__all__ = ["DespikeHampel", "AsLSBaseline"]
