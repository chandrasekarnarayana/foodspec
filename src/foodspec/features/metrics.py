"""Feature stability and discriminative power metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif


def feature_cv(df: pd.DataFrame) -> pd.Series:
    """Compute coefficient of variation per feature.

    Args:
        df: Feature DataFrame (samples × features).

    Returns:
        Series with CV (std/mean) for each feature column.
    """

    return df.std(ddof=1) / (df.mean() + 1e-12)


def feature_stability_by_group(df: pd.DataFrame, groups: Iterable) -> pd.Series:
    """Compute average CV across groups (replicate stability).

    Args:
        df: Feature DataFrame (samples × features).
        groups: Iterable of group labels (length must match df rows).

    Returns:
        Series with mean CV for each feature column across groups.

    Raises:
        ValueError: If groups length does not match df rows.
    """

    groups = list(groups)
    if len(groups) != len(df):
        raise ValueError("groups length must match df rows")
    grouped = df.copy()
    grouped["__grp"] = groups
    cvs: List[pd.Series] = []
    for _, sub in grouped.groupby("__grp"):
        cvs.append(feature_cv(sub.drop(columns="__grp")))
    return pd.concat(cvs, axis=1).mean(axis=1)


def discriminative_power(df: pd.DataFrame, labels: Iterable, n_neighbors: int = 3) -> Dict[str, float]:
    """Compute ANOVA F and mutual information for features.

    Args:
        df: Feature DataFrame (samples × features).
        labels: Iterable of class labels (length must match df rows).
        n_neighbors: Number of neighbors for MI estimation.

    Returns:
        Dictionary with keys 'anova_f_mean' and 'mi_mean' (mean F-statistic
        and mutual information across features).

    Raises:
        ValueError: If labels length does not match df rows.
    """

    y = np.asarray(list(labels))
    if len(y) != len(df):
        raise ValueError("labels length must match df rows")
    X = df.to_numpy()
    f_vals, _ = f_classif(X, y)
    mi = mutual_info_classif(X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=42)
    return {
        "anova_f_mean": float(np.nanmean(f_vals)),
        "mi_mean": float(np.nanmean(mi)),
    }


def robustness_vs_variations(feature_tables: List[pd.DataFrame]) -> float:
    """Measure robustness to preprocessing variation via mean pairwise correlation.

    Args:
        feature_tables: List of feature DataFrames (same columns, different preprocessing).

    Returns:
        Mean pairwise correlation (0.0 to 1.0) across all feature table pairs.
    """

    if len(feature_tables) < 2:
        return 1.0
    corrs = []
    base_cols = feature_tables[0].columns
    for i in range(len(feature_tables)):
        for j in range(i + 1, len(feature_tables)):
            df_i = feature_tables[i][base_cols]
            df_j = feature_tables[j][base_cols]
            corr = df_i.corrwith(df_j, axis=1).mean()
            corrs.append(corr)
    mean_corr = np.nanmean(corrs)
    if np.isnan(mean_corr):
        return 0.0
    return float(mean_corr)


__all__ = [
    "feature_cv",
    "feature_stability_by_group",
    "discriminative_power",
    "robustness_vs_variations",
]
