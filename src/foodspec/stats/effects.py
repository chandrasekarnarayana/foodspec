"""
Effect size and summary utilities.

Provides Cohen's d for two-group comparisons and eta-squared/partial eta-squared
for ANOVA. Designed to pair with hypothesis test outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cohens_d(group1, group2, pooled: bool = True) -> float:
    """
    Compute Cohen's d for two groups.

    Parameters
    ----------
    group1, group2 : array-like
        Samples for the two groups.
    pooled : bool, optional
        If True, use pooled standard deviation; otherwise use unpooled average,
        by default True.

    Returns
    -------
    float
        Cohen's d effect size.
    """

    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    m1, m2 = g1.mean(), g2.mean()
    if pooled:
        s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
        n1, n2 = len(g1), len(g2)
        sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (m1 - m2) / sp
    else:
        s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
        d = (m1 - m2) / ((s1 + s2) / 2.0)
    return float(d)
    __all__ = ["compute_cohens_d", "compute_anova_effect_sizes"]

def compute_anova_effect_sizes(ss_between: float, ss_total: float, ss_within: float | None = None) -> pd.Series:
    """
    Compute eta-squared and partial eta-squared for ANOVA.

    Parameters
    ----------
    ss_between : float
        Sum of squares between groups.
    ss_total : float
        Total sum of squares.
    ss_within : float | None, optional
        Sum of squares within groups (error term). If provided, partial
        eta-squared is computed.

    Returns
    -------
    pd.Series
        eta_squared and partial_eta_squared (if ss_within provided).
    """

    eta_sq = ss_between / ss_total if ss_total != 0 else np.nan
    result = {"eta_squared": eta_sq}
    if ss_within is not None and (ss_between + ss_within) != 0:
        result["partial_eta_squared"] = ss_between / (ss_between + ss_within)
    return pd.Series(result)
