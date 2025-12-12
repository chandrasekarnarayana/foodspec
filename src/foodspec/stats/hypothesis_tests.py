"""
Hypothesis testing utilities for FoodSpec.

Provides wrappers around common tests (t-tests, ANOVA, MANOVA, Tukey HSD)
with simple inputs (arrays/DataFrames or FoodSpectrumSet + column names).
Uses SciPy/statsmodels under the hood to avoid reimplementing core statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    from statsmodels.multivariate.manova import MANOVA
    from statsmodels.stats.multicomp import pairwise_gameshowell, pairwise_tukeyhsd
except ImportError:  # pragma: no cover
    sm = None
    MANOVA = None
    pairwise_tukeyhsd = None
    pairwise_gameshowell = None



@dataclass
class TestResult:
    """Container for hypothesis test outputs."""

    statistic: float
    pvalue: float
    df: Optional[float]
    summary: pd.DataFrame


def _to_series(values) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    return pd.Series(np.asarray(values))


def run_ttest(
    sample1,
    sample2=None,
    popmean: float | None = None,
    paired: bool = False,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Run a t-test (one-sample, two-sample, or paired).

    Parameters
    ----------
    sample1 : array-like
        First sample. For one-sample tests, this is the sample; for paired tests,
        this is the first paired series.
    sample2 : array-like, optional
        Second sample for two-sample or paired tests.
    popmean : float, optional
        Population mean for one-sample test. If provided and sample2 is None,
        a one-sample test is run.
    paired : bool, optional
        If True, performs a paired t-test using sample1 and sample2, by default False.
    alternative : str, optional
        'two-sided', 'less', or 'greater', by default 'two-sided'.

    Returns
    -------
    TestResult
        statistic, pvalue, df, and a summary DataFrame.

    Notes
    -----
    Assumes approximate normality; use nonparametric alternatives if violated.
    """

    s1 = _to_series(sample1)
    df_val = None
    if sample2 is None and popmean is not None:
        stat, p = stats.ttest_1samp(s1, popmean=popmean, alternative=alternative)
        df_val = len(s1) - 1
    elif paired and sample2 is not None:
        s2 = _to_series(sample2)
        stat, p = stats.ttest_rel(s1, s2, alternative=alternative)
        df_val = len(s1) - 1
    elif sample2 is not None:
        s2 = _to_series(sample2)
        stat, p = stats.ttest_ind(s1, s2, equal_var=False, alternative=alternative)
        df_val = len(s1) + len(s2) - 2
    else:
        raise ValueError("Provide popmean for one-sample or sample2 for two-sample/paired tests.")

    summary = pd.DataFrame(
        [{"test": "t-test", "statistic": stat, "pvalue": p, "df": df_val, "alternative": alternative}]
    )
    return TestResult(statistic=float(stat), pvalue=float(p), df=df_val, summary=summary)


def run_anova(data, groups) -> TestResult:
    """
    Run one-way ANOVA on numeric data grouped by labels.

    Parameters
    ----------
    data : array-like
        Numeric values (e.g., peak ratios).
    groups : array-like
        Group labels of same length as data (e.g., oil_type).

    Returns
    -------
    TestResult

    Notes
    -----
    Assumes normality and homogeneity of variance. For a nonparametric alternative, use
    `run_kruskal(data, groups)`.
    """

    df = pd.DataFrame({"data": np.asarray(data), "group": np.asarray(groups)})
    grouped = [grp["data"].to_numpy() for _, grp in df.groupby("group")]
    stat, p = stats.f_oneway(*grouped)
    summary = pd.DataFrame([{"test": "anova_one_way", "statistic": stat, "pvalue": p, "df": None}])
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def run_manova(data: pd.DataFrame, groups: Iterable) -> TestResult:
    """
    Run MANOVA (multivariate ANOVA) if statsmodels is available.

    Parameters
    ----------
    data : pd.DataFrame
        Numeric columns for multivariate response (e.g., multiple ratios/PCs).
    groups : array-like
        Group labels matching rows in data.

    Returns
    -------
    TestResult

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    """

    if MANOVA is None:
        raise ImportError("statsmodels is required for MANOVA.")
    df = data.copy()
    df["_group"] = np.asarray(groups)
    # statsmodels MANOVA uses endog/exog, use from_formula convenience
    mv = MANOVA.from_formula(f"{' + '.join(data.columns)} ~ _group", data=df)
    res = mv.mv_test()
    # Extract Wilks' Lambda row for _group
    tbl = res.results["_group"]["stat"]
    wilks = float(tbl.loc["Wilks' lambda", "Value"])
    pval = float(tbl.loc["Wilks' lambda", "Pr > F"])
    summary = tbl.reset_index()
    summary["test"] = "MANOVA (Wilks)"
    return TestResult(statistic=wilks, pvalue=pval, df=None, summary=summary)


def run_tukey_hsd(values, groups, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run Tukey HSD post-hoc comparisons after ANOVA.

    Parameters
    ----------
    values : array-like
        Numeric data.
    groups : array-like
        Group labels.
    alpha : float, optional
        Significance level, by default 0.05.

    Returns
    -------
    pd.DataFrame
        Pairwise comparisons with mean difference, p-adj, CI, reject flag.

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    """

    if pairwise_tukeyhsd is None:
        raise ImportError("statsmodels is required for Tukey HSD.")
    res = pairwise_tukeyhsd(endog=np.asarray(values), groups=np.asarray(groups), alpha=alpha)
    tbl = pd.DataFrame(
        {
            "group1": res.groupsunique[res._multicomp.pairindices[0]],
            "group2": res.groupsunique[res._multicomp.pairindices[1]],
            "meandiff": res.meandiffs,
            "p_adj": res.pvalues,
            "lower": res.confint[:, 0],
            "upper": res.confint[:, 1],
            "reject": res.reject,
        }
    )
    return tbl


def games_howell(values, groups, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run Games–Howell post-hoc comparisons (robust to unequal variances/sizes).

    Parameters
    ----------
    values : array-like
        Numeric observations (e.g., ratios or peak intensities).
    groups : array-like
        Group labels of the same length as ``values`` (>=2 groups).
    alpha : float, optional
        Significance level, by default 0.05.

    Returns
    -------
    pd.DataFrame
        Pairwise comparisons with mean difference, adjusted p-value, confidence
        intervals, and reject flag. Falls back to a Welch-style computation if
        statsmodels' ``pairwise_gameshowell`` is unavailable.
    """

    vals = np.asarray(values)
    grps = np.asarray(groups)
    if vals.shape[0] != grps.shape[0]:
        raise ValueError("values and groups must have the same length")

    if pairwise_gameshowell is not None:
        res = pairwise_gameshowell(endog=vals, groups=grps, alpha=alpha)
        tbl = pd.DataFrame(
            {
                "group1": res.groupsunique[res._multicomp.pairindices[0]],
                "group2": res.groupsunique[res._multicomp.pairindices[1]],
                "meandiff": res.meandiffs,
                "p_adj": res.pvalues,
                "lower": res.confint[:, 0],
                "upper": res.confint[:, 1],
                "reject": res.reject,
            }
        )
        return tbl

    # Fallback: Welch-style approximation using t distribution
    results: list[Tuple[str, str, float, float, float, float, bool]] = []
    for i, gi in enumerate(np.unique(grps)):
        vi = vals[grps == gi]
        ni = len(vi)
        mi = np.mean(vi)
        si2 = np.var(vi, ddof=1)
        for gj in np.unique(grps)[i + 1 :]:
            vj = vals[grps == gj]
            nj = len(vj)
            mj = np.mean(vj)
            sj2 = np.var(vj, ddof=1)
            diff = mi - mj
            se = np.sqrt(si2 / ni + sj2 / nj)
            t_stat = np.abs(diff) / se
            df_num = (si2 / ni + sj2 / nj) ** 2
            df_den = (si2**2) / (ni**2 * (ni - 1)) + (sj2**2) / (nj**2 * (nj - 1))
            df = df_num / df_den
            p_val = stats.t.sf(t_stat, df) * 2
            # Simple CI using t critical
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            half_width = t_crit * se
            lower = diff - half_width
            upper = diff + half_width
            reject = p_val < alpha
            results.append((gi, gj, diff, p_val, lower, upper, reject))

    return pd.DataFrame(
        results,
        columns=["group1", "group2", "meandiff", "p_adj", "lower", "upper", "reject"],
    )


def run_mannwhitney_u(
    data,
    group_col: str | None = None,
    value_col: str | None = None,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Run Mann–Whitney U test (nonparametric two-sample test).

    Parameters
    ----------
    data : array-like or pd.DataFrame
        If DataFrame, supply `group_col` and `value_col`. If array-like, should be tuple/list of two samples.
    group_col : str, optional
        Column with group labels (exactly two groups).
    value_col : str, optional
        Column with numeric values.
    alternative : str, optional
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    TestResult

    Notes
    -----
    Use when normality/variance assumptions for t-test are doubtful.
    """

    if isinstance(data, pd.DataFrame):
        if group_col is None or value_col is None:
            raise ValueError("Provide group_col and value_col when passing a DataFrame.")
        groups = data[group_col].unique()
        if len(groups) != 2:
            raise ValueError("Mann–Whitney U requires exactly two groups.")
        g1 = data.loc[data[group_col] == groups[0], value_col].to_numpy()
        g2 = data.loc[data[group_col] == groups[1], value_col].to_numpy()
    else:
        # Allow either a tuple/list of two samples, or passing the second sample via
        # `group_col` for convenience in tests or quick calls.
        if group_col is not None and not isinstance(group_col, str) and value_col is None:
            try:
                g1, g2 = data, group_col
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    "Provide two samples (g1, g2) or "
                    "a DataFrame with group/value columns."
                ) from exc
        else:
            try:
                g1, g2 = data
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    "Provide two samples (g1, g2) or "
                    "a DataFrame with group/value columns."
                ) from exc

    stat, p = stats.mannwhitneyu(g1, g2, alternative=alternative)
    summary = pd.DataFrame(
        [
            {
                "test": "mannwhitney_u",
                "statistic": stat,
                "pvalue": p,
                "df": None,
                "alternative": alternative,
            }
        ]
    )
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def run_wilcoxon_signed_rank(sample_before, sample_after, alternative: str = "two-sided") -> TestResult:
    """
    Run Wilcoxon signed-rank test for paired samples (nonparametric).

    Parameters
    ----------
    sample_before : array-like
        First paired measurements.
    sample_after : array-like
        Second paired measurements.
    alternative : str, optional
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    TestResult

    Notes
    -----
    Use when paired t-test assumptions are doubtful (non-normal differences).
    """

    s1 = _to_series(sample_before)
    s2 = _to_series(sample_after)
    stat, p = stats.wilcoxon(s1, s2, alternative=alternative)
    summary = pd.DataFrame(
        [{"test": "wilcoxon_signed_rank", "statistic": stat, "pvalue": p, "df": None, "alternative": alternative}]
    )
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def run_kruskal_wallis(data, group_col: str | None = None, value_col: str | None = None) -> TestResult:
    """
    Run Kruskal–Wallis H-test for independent samples (nonparametric ANOVA).

    Parameters
    ----------
    data : array-like or pd.DataFrame
        If DataFrame, supply `group_col` and `value_col`. If array-like, provide an iterable of groups.
    group_col : str, optional
        Column with group labels (>=2 groups).
    value_col : str, optional
        Column with numeric values.

    Returns
    -------
    TestResult

    Notes
    -----
    Use when ANOVA assumptions (normality/variance) are violated or data are ordinal.
    """

    if isinstance(data, pd.DataFrame):
        if group_col is None or value_col is None:
            raise ValueError("Provide group_col and value_col when passing a DataFrame.")
        grouped = [grp[value_col].to_numpy() for _, grp in data.groupby(group_col)]
    else:
        grouped = [np.asarray(g) for g in data]
    if len(grouped) < 2:
        raise ValueError("Kruskal–Wallis requires at least two groups.")
    stat, p = stats.kruskal(*grouped)
    summary = pd.DataFrame([{"test": "kruskal_wallis", "statistic": stat, "pvalue": p, "df": None}])
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def run_friedman_test(*samples) -> TestResult:
    """
    Run Friedman test for repeated measures (nonparametric).

    Parameters
    ----------
    *samples : array-like
        Repeated measures for each condition (same length per condition).

    Returns
    -------
    TestResult
    """

    stat, p = stats.friedmanchisquare(*samples)
    summary = pd.DataFrame([{"test": "friedman", "statistic": stat, "pvalue": p, "df": None}])
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)
