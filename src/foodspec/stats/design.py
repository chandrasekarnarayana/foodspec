"""
Study design helpers.

Summarize group sizes and flag undersampled designs that may affect tests like
ANOVA/MANOVA. Keep utilities lightweight and metadata-driven.
"""

from __future__ import annotations

import pandas as pd


def summarize_group_sizes(groups) -> pd.Series:
    """
    Summarize counts per group.

    Parameters
    ----------
    groups : array-like or pd.Series
        Group labels.

    Returns
    -------
    pd.Series
        Counts per group.
    """

    return pd.Series(groups).value_counts()


def check_minimum_samples(groups, min_per_group: int = 2) -> pd.DataFrame:
    """
    Check whether each group meets a minimum sample count.

    Parameters
    ----------
    groups : array-like or pd.Series
        Group labels.
    min_per_group : int, optional
        Minimum acceptable samples per group, by default 2.

    Returns
    -------
    pd.DataFrame
        Columns: group, count, ok (bool).
    """

    counts = summarize_group_sizes(groups)
    df = counts.reset_index()
    df.columns = ["group", "count"]
    df["ok"] = df["count"] >= min_per_group
    return df
