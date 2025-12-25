"""
Hypothesis testing utilities for FoodSpec.

Wrappers around common tests (t-test, ANOVA, MANOVA, Tukey HSD, Games–Howell,
nonparametric tests), plus FDR correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:  # Optional imports
    from statsmodels.multivariate.manova import MANOVA
    from statsmodels.stats.multicomp import pairwise_gameshowell, pairwise_tukeyhsd
except Exception:  # pragma: no cover
    MANOVA = None
    pairwise_tukeyhsd = None
    pairwise_gameshowell = None


@dataclass
class TestResult:
    """Immutable result container for hypothesis tests.
    
    Stores test statistics, p-values, degrees of freedom, and summary tables
    from all statistical tests. Designed for reproducibility and regulatory
    compliance (FDA 21 CFR Part 11).
    
    **Interpretation Guide:**
    p-value (pvalue) alone is insufficient. Always check:
    1. **Effect size:** Is the difference practically meaningful?
    2. **Sample size:** Large n → small p even for trivial effects
    3. **Assumptions:** Are test assumptions met? (Normality, equal variance, etc.)
    4. **Multiple testing:** Correct p-value for multiple comparisons (FDR/Bonferroni)
    
    **When to Trust p-value:**
    - p < 0.001: Highly significant, but check effect size
    - p < 0.01: Significant, likely real effect
    - p < 0.05: Marginally significant, interpret with caution
    - p > 0.05: Not significant (≠ proof of no effect; may be underpowered)
    
    Attributes:
        statistic (float): Test statistic (t, F, χ², U, etc.)
        pvalue (float): p-value from test (0–1)
        df (float, optional): Degrees of freedom (if applicable)
        summary (pd.DataFrame): Human-readable results table
        term (str, optional): MANOVA term name (e.g., "group", "time")
    
    See Also:
        - [Metric Significance Tables](../09-reference/metric_significance_tables.md)
    """
    statistic: float
    pvalue: float
    df: Optional[float]
    summary: pd.DataFrame
    term: Optional[str] = None


def _to_series(x) -> pd.Series:
    return pd.Series(np.asarray(x), dtype=float)


def run_ttest(
    sample1,
    sample2: Optional[Iterable] = None,
    popmean: Optional[float] = None,
    paired: bool = False,
    alternative: str = "two-sided",
) -> TestResult:
    """t-test: one-sample, paired, or two-sample (Welch's).
    
    Tests whether sample mean(s) differ significantly from a population mean
    or each other. Welch's t-test (default for two-sample) does NOT assume
    equal variances, making it more robust than Student's t-test.
    
    **Test Selection:**
    - One-sample: `run_ttest(x, popmean=5)` — Does x differ from 5?
    - Paired: `run_ttest(before, after, paired=True)` — Before vs. after?
    - Two-sample: `run_ttest(groupA, groupB)` — Do groups differ?
    
    **Assumptions:** Normality (robust if n > 30), independence, random sampling
    
    **Significance:** See [Metric Significance Tables](../09-reference/metric_significance_tables.md)
    
    Parameters:
        sample1 (array-like): First sample
        sample2 (array-like, optional): Second sample
        popmean (float, optional): Population mean (for one-sample)
        paired (bool): Whether samples are paired
        alternative (str): "two-sided", "less", or "greater"
    
    Returns:
        TestResult with t-statistic, p-value, df
    
    Examples:
        >>> from foodspec.stats import run_ttest
        >>> result = run_ttest([1, 2, 3], [4, 5, 6])\n        >>> print(f"p = {result.pvalue:.3f}")\n    
    See Also:
        - [T-tests & Effect Sizes](../stats/t_tests_effect_sizes_and_power.md)
        - run_mannwhitney_u(): Non-parametric alternative
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
    """One-way ANOVA: test if 3+ groups have different means.
    
    Tests null hypothesis that all group means are equal. ANOVA partitions
    variance into between-group and within-group components. F-statistic is
    their ratio: large F → significant difference.
    
    **When to Use:**
    - 3+ groups (use t-test for 2 groups)
    - Roughly equal sample sizes per group
    - Data approximately normal (robust if n > 20 per group)
    
    **Assumptions:** Normality, homogeneity of variance, independence
    
    **Post-hoc Tests (if p < 0.05):**
    - Balanced design: Tukey HSD (run_tukey_hsd)
    - Unequal variances: Games-Howell (games_howell)
    - Multiple hypotheses: Benjamini-Hochberg FDR (benjamini_hochberg)
    
    **Red Flags:**
    - Significant ANOVA but non-significant post-hoc: likely Type I error
    - Large effect size (η² > 0.14) but p > 0.05: underpowered
    
    Parameters:
        data (array-like): All observations
        groups (array-like): Group labels (same length as data)
    
    Returns:
        TestResult with F-statistic, p-value
    
    Examples:
        >>> from foodspec.stats import run_anova
        >>> result = run_anova([1, 2, 1, 5, 6, 5, 9, 10, 9],
        ...                    ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])\n        >>> assert result.pvalue < 0.05  # Groups differ\n    
    See Also:
        - [ANOVA & MANOVA](../stats/anova_and_manova.md)
        - [Metric Significance Tables](../09-reference/metric_significance_tables.md) — Effect size (η²)
        - run_kruskal_wallis(): Non-parametric alternative
    """
    df = pd.DataFrame({"data": np.asarray(data), "group": np.asarray(groups)})
    grouped = [grp["data"].to_numpy() for _, grp in df.groupby("group")]
    stat, p = stats.f_oneway(*grouped)
    summary = pd.DataFrame([{"test": "anova_one_way", "statistic": stat, "pvalue": p, "df": None}])
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def run_shapiro(values) -> TestResult:
    """Shapiro-Wilk test for normality of data.
    
    Tests whether data comes from a normal distribution. Many parametric
    tests (t-test, ANOVA) assume normality; use this to check assumptions.
    
    **Null Hypothesis:** Data is normally distributed
    
    **Interpretation:**
    - p < 0.05: Data significantly non-normal → consider non-parametric test
    - p > 0.05: Data consistent with normal distribution
    
    **Important:** p-value does NOT guarantee normality (just no evidence against).
    Also, large samples (n > 100) show p < 0.05 even for tiny departures.
    Always pair with visual inspection (Q-Q plot).
    
    **Robust Alternatives:** t-test robust if n > 30 (CLT applies)
    
    **Red Flags:**
    - p-value < 0.05 + clear outliers: outliers causing non-normality
    - p-value < 0.05 + n > 500: minor non-normality (parametric test still OK)
    - p-value > 0.05 + bimodal plot: multimodal data (visual check matters!)
    
    Parameters:
        values (array-like): Data to test
    
    Returns:
        TestResult with Shapiro-Wilk statistic and p-value
    
    Examples:
        >>> from foodspec.stats import run_shapiro
        >>> import numpy as np\n        >>> normal = np.random.normal(100, 10, 30)\n        >>> result = run_shapiro(normal)\n        >>> assert result.pvalue > 0.05  # Normally distributed\n    
    See Also:
        - [Non-parametric Methods](../stats/nonparametric_methods_and_robustness.md)
    """
    vals = np.asarray(values)
    stat, p = stats.shapiro(vals)
    summary = pd.DataFrame([{"test": "shapiro", "statistic": stat, "pvalue": p, "df": None}])
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def benjamini_hochberg(pvalues: Iterable[float], alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction for multiple hypothesis testing.
    
    Controls False Discovery Rate (FDR): expected proportion of false positives
    among rejected hypotheses. Less conservative than Bonferroni, allowing more
    rejections while controlling error rate.
    
    **When to Use:**
    Testing 20+ hypotheses: expect ~1 false positive by chance (α=0.05)
    
    **Comparison:**
    - No correction: All significant at α=0.05 (high false positive rate)
    - Bonferroni: α/m threshold → few rejections but low false positive rate
    - Benjamini-Hochberg: Adaptive threshold → balanced rejections/errors
    
    **Output Interpretation:**
    - p_adj < α: Reject hypothesis (significant after correction)
    - p_adj ≥ α: Fail to reject (not significant)
    - reject: Boolean column (True/False for easy filtering)
    
    **Red Flags:**
    - All p-values > 0.99: Invalid test assumptions or independence violated
    - p-values bimodal (spike at 0 and 1): mixture of true/false positives
    - Very conservative results: consider if α too small
    
    Parameters:
        pvalues (iterable): Raw p-values from multiple tests (0–1)
        alpha (float, default 0.05): Target FDR level
    
    Returns:
        pd.DataFrame with columns: pvalue, p_adj, reject
    
    Examples:
        >>> from foodspec.stats import benjamini_hochberg\n        >>> pvalues = [0.001, 0.01, 0.03, 0.05, 0.1, 0.5] * 3\n        >>> result = benjamini_hochberg(pvalues, alpha=0.05)\n        >>> significant = result[result['reject']]\n        >>> print(f"Significant: {len(significant)}/{len(result)}")\n    
    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: a practical and powerful approach to multiple testing.
    
    See Also:
        - [Metric Significance Tables](../09-reference/metric_significance_tables.md) — Multiple testing
    """
    pvals = np.asarray(list(pvalues), dtype=float)
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return pd.DataFrame({"pvalue": pvals, "p_adj": p_adj, "reject": reject})


def run_manova(
    df: pd.DataFrame,
    group_col: Optional[object] = None,
    dependent_cols: Optional[List[str]] = None,
    groups: Optional[Iterable] = None,
) -> TestResult:
    """
    MANOVA via statsmodels.

    Supports two usage patterns:
    - run_manova(df, group_col="group", dependent_cols=["f1","f2"])  # formula
    - run_manova(data_df, groups=labels)  # use all columns in data_df

    Returns a TestResult with `pvalue` extracted from the MANOVA table
    (prefers Wilks' lambda, falls back to Pillai's trace).
    """
    if MANOVA is None:
        from statsmodels.multivariate.manova import MANOVA as _MANOVA  # type: ignore

        globals()["MANOVA"] = _MANOVA

    # Case 1: group_col is a column name in df (formula interface)
    term_name = None
    if group_col is not None and isinstance(group_col, str):
        cols = dependent_cols or [c for c in df.columns if c != group_col]
        formula = " + ".join(list(cols)) + " ~ " + group_col
        mv = MANOVA.from_formula(formula, data=df)
        term_name = group_col
    else:
        # Case 2: groups provided explicitly or group_col is an array-like of labels
        data = df.copy()
        grp = np.asarray(groups if groups is not None else group_col)
        if grp is None:
            raise ValueError("Provide (group_col, dependent_cols) or groups with data df.")
        data["_group"] = grp
        dep_cols = dependent_cols or list(df.columns)
        formula = " + ".join(dep_cols) + " ~ _group"
        mv = MANOVA.from_formula(formula, data=data)
        term_name = "_group"

    res = mv.mv_test()

    # Extract p-value and an associated F statistic for the grouping term
    # statsmodels structure: res.results[term_name]["stat"] is a DataFrame with
    # rows like "Wilks' lambda", "Pillai's trace" and columns including
    # "F Value" and "Pr > F".
    results_map = getattr(res, "results", None)
    if not isinstance(results_map, dict) or not results_map:
        # Fallback: create an empty TestResult
        return TestResult(statistic=np.nan, pvalue=np.nan, df=None, summary=pd.DataFrame())

    # Resolve the term key to use
    if term_name not in results_map:
        # Try the first non-intercept term if available
        candidate_keys = [k for k in results_map.keys() if str(k).lower() != "intercept"]
        term_key = candidate_keys[0] if candidate_keys else list(results_map.keys())[0]
    else:
        term_key = term_name

    stat_df = results_map[term_key].get("stat")
    if not isinstance(stat_df, pd.DataFrame):
        return TestResult(statistic=np.nan, pvalue=np.nan, df=None, summary=pd.DataFrame())

    # Prefer Wilks' lambda; fall back to Pillai's trace if not present
    def _normalize_idx_name(s: str) -> str:
        return s.lower().replace("'", "").replace("-", "").replace(" ", "")

    preferred = ["wilkslambda", "pillaistrace"]
    chosen_row = None
    for idx in stat_df.index:
        norm = _normalize_idx_name(str(idx))
        if norm in preferred:
            chosen_row = idx
            break
    if chosen_row is None:
        # Use the first row as a last resort
        chosen_row = stat_df.index[0]

    # Extract p-value and F statistic
    try:
        pval = float(stat_df.loc[chosen_row, "Pr > F"])  # type: ignore[index]
    except Exception:
        # Some versions might use lowercase or a different label
        pval = float(stat_df.loc[chosen_row].filter(like="Pr").iloc[0])
    try:
        fval = float(stat_df.loc[chosen_row, "F Value"])  # type: ignore[index]
    except Exception:
        fval = np.nan

    # Return TestResult for consistency with other functions; include term name
    return TestResult(statistic=fval, pvalue=pval, df=None, summary=stat_df, term=str(term_key))


def run_tukey_hsd(values, groups, alpha: float = 0.05) -> pd.DataFrame:
    if pairwise_tukeyhsd is None:
        raise ImportError("statsmodels is required for Tukey HSD.")
    res = pairwise_tukeyhsd(endog=np.asarray(values), groups=np.asarray(groups), alpha=alpha)
    return pd.DataFrame(
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


def games_howell(values, groups, alpha: float = 0.05) -> pd.DataFrame:
    vals = np.asarray(values)
    grps = np.asarray(groups)
    if vals.shape[0] != grps.shape[0]:
        raise ValueError("values and groups must have the same length")
    if pairwise_gameshowell is not None:
        res = pairwise_gameshowell(endog=vals, groups=grps, alpha=alpha)
        return pd.DataFrame(
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
    # Welch-style fallback
    rows = []
    uniq = np.unique(grps)
    for i, gi in enumerate(uniq):
        vi = vals[grps == gi]
        ni = len(vi)
        mi = np.mean(vi)
        si2 = np.var(vi, ddof=1)
        for gj in uniq[i + 1 :]:
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
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            half = t_crit * se
            rows.append((gi, gj, diff, p_val, diff - half, diff + half, p_val < alpha))
    return pd.DataFrame(rows, columns=["group1", "group2", "meandiff", "p_adj", "lower", "upper", "reject"])


def run_mannwhitney_u(
    data, group_col: object | None = None, value_col: str | None = None, alternative: str = "two-sided"
) -> TestResult:
    if isinstance(data, pd.DataFrame):
        if group_col is None or value_col is None:
            raise ValueError("Provide group_col and value_col when passing a DataFrame.")
        groups = data[group_col].unique()
        if len(groups) != 2:
            raise ValueError("Mann–Whitney U requires exactly two groups.")
        g1 = data.loc[data[group_col] == groups[0], value_col].to_numpy()
        g2 = data.loc[data[group_col] == groups[1], value_col].to_numpy()
    else:
        # Support two calling styles:
        # 1) run_mannwhitney_u([x1,...], [x2,...])
        # 2) run_mannwhitney_u([x1,...], group_col=[x2,...])
        if group_col is not None and value_col is None and hasattr(group_col, "__iter__"):
            g1 = np.asarray(data, dtype=float)
            g2 = np.asarray(group_col, dtype=float)
        else:
            try:
                g1, g2 = data
            except Exception as exc:
                raise ValueError(
                    "Provide either a DataFrame with group/value columns, two arrays, or call with (array1, array2)."
                ) from exc
    stat, p = stats.mannwhitneyu(g1, g2, alternative=alternative)
    summary = pd.DataFrame(
        [{"test": "mannwhitney_u", "statistic": stat, "pvalue": p, "df": None, "alternative": alternative}]
    )
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def run_wilcoxon_signed_rank(sample_before, sample_after, alternative: str = "two-sided") -> TestResult:
    s1 = _to_series(sample_before)
    s2 = _to_series(sample_after)
    stat, p = stats.wilcoxon(s1, s2, alternative=alternative)
    summary = pd.DataFrame(
        [{"test": "wilcoxon_signed_rank", "statistic": stat, "pvalue": p, "df": None, "alternative": alternative}]
    )
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def run_kruskal_wallis(data, group_col: str | None = None, value_col: str | None = None) -> TestResult:
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
    stat, p = stats.friedmanchisquare(*samples)
    summary = pd.DataFrame([{"test": "friedman", "statistic": stat, "pvalue": p, "df": None}])
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


# Convenience alias expected by some users/tests
def tukey_hsd(values, groups, alpha: float = 0.05):
    return run_tukey_hsd(values, groups, alpha=alpha)
