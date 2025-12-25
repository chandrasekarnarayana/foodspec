"""
Effect size and summary utilities.

Provides Cohen's d for two-group comparisons and eta-squared/partial eta-squared
for ANOVA. Designed to pair with hypothesis test outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cohens_d(group1, group2, pooled: bool = True) -> float:
    """Compute Cohen's d effect size for comparing two groups.
    
    Quantifies the magnitude of difference between two groups independent of
    sample size. Essential for evaluating whether statistically significant
    differences are also practically significant.
    
    Parameters:
        group1 (array-like): First group numerical samples for comparison
        group2 (array-like): Second group numerical samples for comparison
        pooled (bool): Use pooled std (assume equal variances). If False, average unpooled std.
    
    Returns:
        float: Cohen's d effect size (unbounded; can be negative)
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


def compute_anova_effect_sizes(ss_between: float, ss_total: float, ss_within: float | None = None) -> pd.Series:
    """Compute eta-squared and partial eta-squared for ANOVA.

    Quantifies proportion of variance explained by group differences.
    
    Interpretation Scale:
    - eta-squared < 0.01: Negligible effect
    - 0.01 <= eta-squared < 0.06: Small effect
    - 0.06 <= eta-squared < 0.14: Medium effect
    - eta-squared >= 0.14: Large effect
    
    Parameters:
        ss_between: Sum of squares between groups (treatment effect)
        ss_total: Total sum of squares (all variation in data)
        ss_within: Sum of squares within groups (error). If provided, partial
            eta-squared is computed.

    Returns:
        pd.Series with 'eta_squared' and optionally 'partial_eta_squared'
    """

    eta_sq = ss_between / ss_total if ss_total != 0 else np.nan
    result = {"eta_squared": eta_sq}
    if ss_within is not None and (ss_between + ss_within) != 0:
        result["partial_eta_squared"] = ss_between / (ss_between + ss_within)
    return pd.Series(result)
