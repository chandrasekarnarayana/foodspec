"""Baseline correction wrappers (engine namespace)."""
from __future__ import annotations

import numpy as np

from foodspec.data_objects.spectral_dataset import (
    baseline_als as _baseline_als,
)
from foodspec.data_objects.spectral_dataset import (
    baseline_polynomial as _baseline_polynomial,
)
from foodspec.data_objects.spectral_dataset import (
    baseline_rubberband as _baseline_rubberband,
)


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return _baseline_als(y, lam=lam, p=p, niter=niter)
    return np.vstack([_baseline_als(row, lam=lam, p=p, niter=niter) for row in y])


def baseline_polynomial(y: np.ndarray, degree: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return _baseline_polynomial(y, degree=degree)
    return np.vstack([_baseline_polynomial(row, degree=degree) for row in y])


def baseline_rubberband(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return _baseline_rubberband(x, y)
    return np.vstack([_baseline_rubberband(x, row) for row in y])


__all__ = ["baseline_als", "baseline_polynomial", "baseline_rubberband"]
