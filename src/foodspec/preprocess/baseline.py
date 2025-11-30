"""Baseline correction transformers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["ALSBaseline", "RubberbandBaseline", "PolynomialBaseline"]


class ALSBaseline(BaseEstimator, TransformerMixin):
    """Asymmetric Least Squares baseline correction (Eilers, 2005)."""

    def __init__(self, lambda_: float = 1e5, p: float = 0.001, max_iter: int = 10):
        self.lambda_ = lambda_
        self.p = p
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ALSBaseline":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        n_samples, n_wavenumbers = X.shape
        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")
        if not (0 < self.p < 1):
            raise ValueError("p must be in (0, 1).")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        D = _second_derivative_matrix(n_wavenumbers)
        baselines = np.zeros_like(X)
        for i, y in enumerate(X):
            w = np.ones(n_wavenumbers)
            for _ in range(self.max_iter):
                W = diags(w, 0, shape=(n_wavenumbers, n_wavenumbers))
                Z = W + self.lambda_ * (D.T @ D)
                z = spsolve(Z, w * y)
                w = self.p * (y > z) + (1 - self.p) * (y < z)
            baseline = z
            corrected_candidate = y - baseline
            edge_mean = corrected_candidate[: min(20, n_wavenumbers)].mean()
            if not np.isfinite(baseline).all() or abs(edge_mean) > 1.0:
                baseline = _poly_baseline(y, degree=2)
            baselines[i, :] = baseline
        return X - baselines


class RubberbandBaseline(BaseEstimator, TransformerMixin):
    """Baseline correction using convex hull (rubberband) approach."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RubberbandBaseline":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")

        n_samples, n_wavenumbers = X.shape
        wavenumbers = np.arange(n_wavenumbers)
        corrected = np.zeros_like(X)

        for i, y in enumerate(X):
            lower = _lower_hull_indices(wavenumbers, y)
            baseline = np.interp(wavenumbers, wavenumbers[lower], y[lower])
            corrected[i, :] = y - baseline

        return corrected


class PolynomialBaseline(BaseEstimator, TransformerMixin):
    """Baseline correction by polynomial fitting."""

    def __init__(self, degree: int = 3):
        self.degree = degree

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PolynomialBaseline":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        if self.degree < 0:
            raise ValueError("degree must be non-negative.")

        n_samples, n_wavenumbers = X.shape
        x_axis = np.linspace(0, 1, n_wavenumbers)
        corrected = np.zeros_like(X)
        for i, y in enumerate(X):
            coefs = np.polyfit(x_axis, y, deg=self.degree)
            baseline = np.polyval(coefs, x_axis)
            corrected[i, :] = y - baseline
        return corrected


def _second_derivative_matrix(n: int) -> csc_matrix:
    diagonals = [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)]
    offsets = [0, 1, 2]
    return diags(diagonals, offsets, shape=(n - 2, n))


def _poly_baseline(y: np.ndarray, degree: int = 2) -> np.ndarray:
    x_axis = np.linspace(0, 1, y.shape[0])
    coefs = np.polyfit(x_axis, y, deg=degree)
    return np.polyval(coefs, x_axis)


def _lower_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute lower hull indices using monotone chain."""

    order = np.argsort(x)
    lower: list[int] = []

    def cross(o: int, a: int, b: int) -> float:
        return (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o])

    for idx in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], idx) <= 0:
            lower.pop()
        lower.append(idx)

    return np.array(lower, dtype=int)
