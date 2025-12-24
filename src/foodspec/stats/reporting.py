"""Statistical reporting helpers for per-feature/group summaries."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from foodspec.stats.hypothesis_tests import (
    run_ttest,
    run_anova,
    run_kruskal_wallis,
    run_mannwhitney_u,
    benjamini_hochberg,
    run_shapiro,
)
from foodspec.stats.effects import compute_cohens_d
from foodspec.stats.correlations import compute_correlations


def stats_report_for_feature(
    values: Iterable[float],
    groups: Iterable,
    *,
    normality: bool = True,
    alpha: float = 0.05,
    feature_name: str = "feature",
    nonparametric: bool | None = None,
) -> pd.DataFrame:
    """Generate a stats report for one feature across groups.

    Chooses ANOVA + t-tests (parametric) or Kruskal + Mann–Whitney (nonparametric)
    based on normality flag. Applies BH correction to pairwise p-values.
    """

    vals = np.asarray(values, dtype=float)
    grps = np.asarray(groups)
    uniq = np.unique(grps)
    rows = []

    # Normality check (overall pooled)
    normal_p = np.nan
    if normality:
        try:
            normal_p = run_shapiro(vals).pvalue
        except Exception:
            normal_p = np.nan

    # Decide test family
    use_nonparam = nonparametric if nonparametric is not None else (normal_p < alpha if normality else False)

    if len(uniq) == 2:
        g1 = vals[grps == uniq[0]]
        g2 = vals[grps == uniq[1]]
        if use_nonparam:
            res = run_mannwhitney_u((g1, g2))
            test_name = "mannwhitney_u"
        else:
            res = run_ttest(g1, g2)
            test_name = "t_test"
        d = compute_cohens_d(g1, g2)
        rows.append(
            {
                "feature": feature_name,
                "test": test_name,
                "statistic": res.statistic,
                "pvalue": res.pvalue,
                "effect_size_d": d,
                "normality_p": normal_p,
                "nonparametric": use_nonparam,
            }
        )
    elif len(uniq) > 2:
        if use_nonparam:
            res = run_kruskal_wallis(vals, group_col=None, value_col=None)
            test_name = "kruskal_wallis"
        else:
            res = run_anova(vals, grps)
            test_name = "anova"
        rows.append(
            {
                "feature": feature_name,
                "test": test_name,
                "statistic": res.statistic,
                "pvalue": res.pvalue,
                "effect_size_d": np.nan,
                "normality_p": normal_p,
                "nonparametric": use_nonparam,
            }
        )
    else:
        raise ValueError("Need at least two groups for statistical comparison.")

    df = pd.DataFrame(rows)
    df["p_adj"] = benjamini_hochberg(df["pvalue"], alpha=alpha)["p_adj"].values
    df["reject"] = df["p_adj"] < alpha
    return df


def stats_report_for_features_table(
    data: pd.DataFrame,
    group_col: str,
    *,
    alpha: float = 0.05,
    normality: bool = True,
    nonparametric: bool | None = None,
) -> pd.DataFrame:
    """Generate a per-feature stats report for all numeric columns vs a group column."""

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != group_col]
    reports = []
    for col in num_cols:
        reports.append(
            stats_report_for_feature(
                data[col],
                data[group_col],
                normality=normality,
                alpha=alpha,
                feature_name=col,
                nonparametric=nonparametric,
            )
        )
    return pd.concat(reports, ignore_index=True)


__all__ = ["stats_report_for_feature", "stats_report_for_features_table"]
