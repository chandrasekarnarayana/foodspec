"""
Correlation and mapping utilities.

Provides Pearson/Spearman correlations and simple cross-correlation for
time-based sequences. Accepts arrays/DataFrames or FoodSpectrumSet-derived
features.
"""

from __future__ import annotations

__all__ = ["compute_correlations", "compute_correlation_matrix", "compute_cross_correlation"]

import numpy as np
import pandas as pd
from scipy import signal, stats


def compute_correlations(data: pd.DataFrame, cols: tuple | list, method: str = "pearson") -> pd.Series:
    """Compute correlation between columns in a DataFrame.

    Args:
        data: DataFrame containing the columns of interest (e.g., ratios vs quality metric).
        cols: Two column names to correlate (tuple or list).
        method: 'pearson' or 'spearman', by default 'pearson'.

    Returns:
        Series with index ['r', 'pvalue']; values are correlation coefficient and p-value.

    Raises:
        ValueError: If cols does not contain exactly two column names.
        ValueError: If method is not 'pearson' or 'spearman'.
    """

    if len(cols) != 2:
        raise ValueError("cols must contain exactly two column names.")
    x = data[cols[0]].to_numpy()
    y = data[cols[1]].to_numpy()
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    return pd.Series({"r": r, "pvalue": p})


def compute_correlation_matrix(data: pd.DataFrame, cols, method: str = "pearson") -> pd.DataFrame:
    """Compute a correlation matrix for selected columns.

    Args:
        data: DataFrame containing columns of interest.
        cols: List of column names to include.
        method: 'pearson' or 'spearman', by default 'pearson'.

    Returns:
        Correlation matrix (DataFrame).

    Raises:
        ValueError: If method is not 'pearson' or 'spearman'.
    """

    subset = data[cols]
    if method == "pearson":
        return subset.corr(method="pearson")
    if method == "spearman":
        return subset.corr(method="spearman")
    raise ValueError("method must be 'pearson' or 'spearman'.")


def compute_cross_correlation(seq1, seq2, max_lag: int = 10) -> pd.DataFrame:
    """Compute cross-correlation between two sequences (e.g., time series of ratios).

    Args:
        seq1: First input sequence (array-like).
        seq2: Second input sequence (array-like).
        max_lag: Maximum lag (both positive and negative) to compute, by default 10.

    Returns:
        DataFrame with columns: lag, correlation.

    Raises:
        ValueError: If seq1 and seq2 do not have the same length.
    """

    x = np.asarray(seq1)
    y = np.asarray(seq2)
    if x.shape[0] != y.shape[0]:
        raise ValueError("seq1 and seq2 must have the same length.")
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            corr = signal.correlate(x[-lag:], y[: lag if lag != 0 else None], mode="valid")[0]
        elif lag > 0:
            corr = signal.correlate(x[:-lag], y[lag:], mode="valid")[0]
        else:
            corr = signal.correlate(x, y, mode="valid")[0]
        # Normalize by length
        corr = corr / (np.linalg.norm(x) * np.linalg.norm(y) + np.finfo(float).eps)
        corrs.append(corr)
    return pd.DataFrame({"lag": lags, "correlation": corrs})
