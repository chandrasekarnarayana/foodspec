"""
Study design helpers.

Summarize group sizes and flag undersampled designs that may affect tests like
ANOVA/MANOVA. Keep utilities lightweight and metadata-driven.
"""

from __future__ import annotations

import pandas as pd


def summarize_group_sizes(groups) -> pd.Series:
    """Summarize counts per group.

    Args:
        groups: Array-like or Series of group labels.

    Returns:
        A Series with counts per group.

    Examples:
        >>> from foodspec.stats.design import summarize_group_sizes
        >>> import numpy as np
        >>> groups = np.array(['A', 'A', 'B', 'B', 'B'])
        >>> counts = summarize_group_sizes(groups)
        >>> counts['B']
        3
    """

    return pd.Series(groups).value_counts()


def check_minimum_samples(groups, min_per_group: int = 2) -> pd.DataFrame:
    """Check whether each group meets a minimum sample count.

    Args:
        groups: Array-like or Series of group labels.
        min_per_group: Minimum acceptable samples per group (default 2).

    Returns:
        A DataFrame with columns: 'group', 'count', 'ok' (bool).

    Examples:
        >>> from foodspec.stats.design import check_minimum_samples
        >>> import numpy as np
        >>> groups = np.array(['A', 'A', 'B'])
        >>> result = check_minimum_samples(groups, min_per_group=2)
        >>> result.loc[result['group'] == 'A', 'ok'].iloc[0]
        True
    """

    counts = summarize_group_sizes(groups)
    df = counts.reset_index()
    df.columns = ["group", "count"]
    df["ok"] = df["count"] >= min_per_group
    return df
