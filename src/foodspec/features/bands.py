"""Band integration utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["integrate_bands"]


def integrate_bands(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    bands: Sequence[Tuple[str, float, float]],
) -> pd.DataFrame:
    """Integrate intensity over specified bands.

    Parameters
    ----------
    X :
        Array of shape (n_samples, n_wavenumbers).
    wavenumbers :
        1D array of wavenumbers aligned with ``X``.
    bands :
        Sequence of (label, min_wn, max_wn) tuples.

    Returns
    -------
    pandas.DataFrame
        Columns named by band labels; one row per sample.
    """

    X = np.asarray(X, dtype=float)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if wavenumbers.ndim != 1 or wavenumbers.shape[0] != X.shape[1]:
        raise ValueError("wavenumbers must be 1D and match number of columns in X.")

    data = {}
    for label, min_wn, max_wn in bands:
        if min_wn >= max_wn:
            raise ValueError(f"Band {label} has invalid range.")
        mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
        if not np.any(mask):
            data[label] = np.full(X.shape[0], np.nan)
            continue
        data[label] = np.trapezoid(X[:, mask], x=wavenumbers[mask], axis=1)

    return pd.DataFrame(data)
