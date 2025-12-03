"""Troubleshooting utilities for quick diagnostics."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def estimate_snr(spectrum: Sequence[float]) -> float:
    """Estimate a crude signal-to-noise ratio for one spectrum.

    Parameters
    ----------
    spectrum : Sequence[float]
        Intensity values for a single spectrum.

    Returns
    -------
    float
        Ratio of (signal std) / (noise std) using a simple median absolute deviation split.

    Notes
    -----
    This is a heuristic; use for relative comparisons, not absolute SNR claims.
    """
    arr = np.asarray(spectrum, dtype=float)
    signal_std = np.std(arr)
    noise_std = np.median(np.abs(arr - np.median(arr))) / 0.6745
    if noise_std == 0:
        return np.inf
    return float(signal_std / noise_std)


def summarize_class_balance(labels: Sequence) -> pd.Series:
    """Summarize class counts.

    Parameters
    ----------
    labels : Sequence
        Iterable of class labels.

    Returns
    -------
    pandas.Series
        Counts per class.
    """
    return pd.Series(labels).value_counts()


def detect_outliers(X: np.ndarray, n_components: int = 5, z_thresh: float = 3.0) -> np.ndarray:
    """Flag potential outliers using PCA score distance.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features).
    n_components : int
        Number of PCA components to use.
    z_thresh : float
        Z-score threshold on score distances.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates a potential outlier.
    """
    pca = PCA(n_components=min(n_components, X.shape[1]))
    scores = pca.fit_transform(X)
    dists = np.linalg.norm(scores, axis=1)
    z = (dists - np.mean(dists)) / (np.std(dists) + 1e-12)
    return np.abs(z) > z_thresh


def check_missing_metadata(df: pd.DataFrame, required_cols: Iterable[str]) -> List[str]:
    """Check for missing required metadata columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table.
    required_cols : Iterable[str]
        Required column names.

    Returns
    -------
    List[str]
        List of missing columns (empty if none).
    """
    missing = [c for c in required_cols if c not in df.columns]
    return missing
