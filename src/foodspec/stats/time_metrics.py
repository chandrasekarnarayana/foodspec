from __future__ import annotations

from typing import Tuple

import numpy as np


def linear_slope(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.shape[0] != y.shape[0]:
        raise ValueError("t and y must be 1D arrays of equal length.")
    coeffs = np.polyfit(t, y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    return slope, intercept


def quadratic_acceleration(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.shape[0] != y.shape[0]:
        raise ValueError("t and y must be 1D arrays of equal length.")
    coeffs = np.polyfit(t, y, 2)
    a = float(coeffs[0])
    return 2.0 * a


def rolling_slope(t: np.ndarray, y: np.ndarray, window: int = 5) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if window < 2:
        raise ValueError("window must be >= 2.")
    n = t.shape[0]
    out = np.full(n, np.nan)
    for i in range(n - window + 1):
        sl, _ = linear_slope(t[i : i + window], y[i : i + window])
        out[i + window - 1] = sl
    return out


__all__ = ["linear_slope", "quadratic_acceleration", "rolling_slope"]
