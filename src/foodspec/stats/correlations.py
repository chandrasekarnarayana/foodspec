"""
Correlation and mapping utilities.

Provides Pearson/Spearman correlations and simple cross-correlation for
time-based sequences. Accepts arrays/DataFrames or FoodSpectrumSet-derived
features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats


def compute_correlations(data: pd.DataFrame, cols, method: str = "pearson") -> pd.Series:
    """
    Compute correlation between columns in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing the columns of interest (e.g., ratios vs quality metric).
    cols : tuple or list
        Two column names to correlate.
    method : str, optional
        'pearson' or 'spearman', by default 'pearson'.

    Returns
    -------
    pd.Series
        index: ['r', 'pvalue']; values are correlation coefficient and p-value.
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
    """
    Compute a correlation matrix for selected columns.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing columns of interest.
    cols : list
        Column names to include.
    method : str, optional
        'pearson' or 'spearman', by default 'pearson'.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """

    subset = data[cols]
    if method == "pearson":
        return subset.corr(method="pearson")
    if method == "spearman":
        return subset.corr(method="spearman")
    raise ValueError("method must be 'pearson' or 'spearman'.")


def compute_cross_correlation(seq1, seq2, max_lag: int = 10) -> pd.DataFrame:
    """
    Compute cross-correlation between two sequences (e.g., time series of ratios).

    Parameters
    ----------
    seq1, seq2 : array-like
        Input sequences of equal length.
    max_lag : int, optional
        Maximum lag (both positive and negative) to compute, by default 10.

    Returns
    -------
    pd.DataFrame
        Columns: lag, correlation.
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
