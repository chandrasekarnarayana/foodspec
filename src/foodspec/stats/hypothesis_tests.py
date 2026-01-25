from __future__ import annotations
"""
Hypothesis testing utilities for FoodSpec.

Wrappers around common tests (t-test, ANOVA, MANOVA, Tukey HSD, Games–Howell,
nonparametric tests), plus FDR correction.
"""


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
        - [Metric Significance Tables](../reference/metric_significance_tables.md)
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

    **Significance:** See [Metric Significance Tables](../reference/metric_significance_tables.md)

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
        - [T-tests & Effect Sizes](../methods/statistics/t_tests_effect_sizes_and_power.md)
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
        - [ANOVA & MANOVA](../methods/statistics/anova_and_manova.md)
        - [Metric Significance Tables](../reference/metric_significance_tables.md) — Effect size (η²)
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
        - [Non-parametric Methods](../methods/statistics/nonparametric_methods_and_robustness.md)
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
        - [Metric Significance Tables](../reference/metric_significance_tables.md) — Multiple testing
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
    """Perform Tukey's Honestly Significant Difference (HSD) post-hoc test.
    
    Tukey's HSD is a multiple comparison test used after ANOVA to identify which
    specific group means are significantly different from each other. It controls
    the family-wise error rate and is appropriate when comparing all pairwise
    combinations of group means.
    
    Args:
        values: Array-like of continuous measurements across all groups.
        groups: Array-like of group labels corresponding to each value.
        alpha: Significance level for confidence intervals (default: 0.05).
    
    Returns:
        DataFrame with columns:
            - group1, group2: The two groups being compared
            - meandiff: Difference in means (group1 - group2)
            - p_adj: Adjusted p-value (Tukey correction)
            - lower, upper: Confidence interval bounds
            - reject: Boolean indicating if null hypothesis is rejected
    
    Raises:
        ImportError: If statsmodels is not installed.
    
    Example:
        >>> from foodspec import run_tukey_hsd
        >>> values = [10, 12, 11, 20, 22, 21, 30, 32, 31]
        >>> groups = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        >>> results = run_tukey_hsd(values, groups)
        >>> print(results)
    
    See Also:
        run_anova: One-way ANOVA test (use before Tukey HSD)
        games_howell: Alternative when variances are unequal
        run_ttest: Pairwise t-tests (less conservative)
    """
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
    data: pd.DataFrame | Iterable,
    group_col: object | None = None,
    value_col: str | None = None,
    alternative: str = "two-sided",
) -> TestResult:
    """Perform Mann-Whitney U test for comparing two independent samples.
    
    The Mann-Whitney U test (also called Wilcoxon rank-sum test) is a non-parametric
    test for comparing the distributions of two independent groups. It does not assume
    normality and is robust to outliers. Use this when data are ordinal or continuous
    but violate t-test assumptions.
    
    Args:
        data: Either a DataFrame or array-like. If DataFrame, must provide group_col
            and value_col. If array-like, represents first sample (requires group_col
            as second sample or unpacking into two samples).
        group_col: Column name for group labels (if data is DataFrame) or second
            sample array (if data is array-like).
        value_col: Column name for values to compare (required if data is DataFrame).
        alternative: Direction of test - "two-sided", "less", or "greater" (default: "two-sided").
    
    Returns:
        TestResult with:
            - statistic: Mann-Whitney U statistic
            - pvalue: Two-tailed (or one-tailed) p-value
            - df: None (non-parametric test has no degrees of freedom)
            - summary: DataFrame with test details
    
    Raises:
        ValueError: If data format is invalid or if not exactly two groups provided.
    
    Example:
        >>> from foodspec import run_mannwhitney_u
        >>> import pandas as pd
        >>> df = pd.DataFrame({'group': ['A']*10 + ['B']*10, 'value': range(20)})
        >>> result = run_mannwhitney_u(df, 'group', 'value')
        >>> print(f"U={result.statistic:.2f}, p={result.pvalue:.4f}")
    
    See Also:
        run_ttest: Parametric alternative for normal data
        run_kruskal_wallis: Extension to 3+ groups
        run_wilcoxon_signed_rank: For paired samples
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
    """Perform Wilcoxon signed-rank test for paired samples.
    
    The Wilcoxon signed-rank test is a non-parametric alternative to the paired t-test.
    It tests whether the median difference between paired observations is zero. Use this
    for repeated measurements or matched pairs when data are non-normal or ordinal.
    
    Args:
        sample_before: Array-like of measurements in the first condition (e.g., before treatment).
        sample_after: Array-like of measurements in the second condition (e.g., after treatment).
            Must have the same length as sample_before.
        alternative: Direction of test - "two-sided", "less", or "greater" (default: "two-sided").
    
    Returns:
        TestResult with:
            - statistic: Wilcoxon signed-rank statistic
            - pvalue: Two-tailed (or one-tailed) p-value
            - df: None (non-parametric test has no degrees of freedom)
            - summary: DataFrame with test details
    
    Example:
        >>> from foodspec import run_wilcoxon_signed_rank
        >>> before = [10, 12, 11, 13, 15, 14, 16, 18]
        >>> after = [11, 13, 12, 15, 16, 15, 18, 20]
        >>> result = run_wilcoxon_signed_rank(before, after)
        >>> print(f"W={result.statistic:.1f}, p={result.pvalue:.4f}")
    
    See Also:
        run_ttest: Parametric paired t-test for normal data
        run_mannwhitney_u: For independent (unpaired) samples
        run_friedman_test: Extension to 3+ repeated measures
    """
    s1 = _to_series(sample_before)
    s2 = _to_series(sample_after)
    stat, p = stats.wilcoxon(s1, s2, alternative=alternative)
    summary = pd.DataFrame(
        [{"test": "wilcoxon_signed_rank", "statistic": stat, "pvalue": p, "df": None, "alternative": alternative}]
    )
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


def run_kruskal_wallis(
    data: pd.DataFrame | Iterable, group_col: str | None = None, value_col: str | None = None
) -> TestResult:
    """Perform Kruskal-Wallis H-test for comparing three or more independent groups.
    
    The Kruskal-Wallis test is a non-parametric alternative to one-way ANOVA. It tests
    whether samples from different groups come from the same distribution. Use this when
    comparing 3+ groups with non-normal data or unequal variances.
    
    Args:
        data: Either a DataFrame or list of arrays. If DataFrame, must provide group_col
            and value_col. If list, each element is an array of values for one group.
        group_col: Column name for group labels (required if data is DataFrame).
        value_col: Column name for values to compare (required if data is DataFrame).
    
    Returns:
        TestResult with:
            - statistic: Kruskal-Wallis H statistic
            - pvalue: p-value from chi-squared distribution
            - df: None (df implicit in chi-squared approximation)
            - summary: DataFrame with test details
    
    Raises:
        ValueError: If fewer than 2 groups provided or if DataFrame arguments are missing.
    
    Example:
        >>> from foodspec import run_kruskal_wallis
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'group': ['A']*10 + ['B']*10 + ['C']*10,
        ...     'value': list(range(10)) + list(range(10, 20)) + list(range(20, 30))
        ... })
        >>> result = run_kruskal_wallis(df, 'group', 'value')
        >>> print(f"H={result.statistic:.2f}, p={result.pvalue:.4f}")
    
    See Also:
        run_anova: Parametric alternative for normal data
        run_mannwhitney_u: For comparing exactly 2 groups
        run_friedman_test: For repeated measures (paired data)
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
    """Perform Friedman test for repeated measures across three or more conditions.
    
    The Friedman test is a non-parametric alternative to repeated measures ANOVA. It tests
    whether there are differences across multiple related (paired) groups. Use this for
    repeated measurements or matched subjects when data are non-normal or ordinal.
    
    Args:
        *samples: Variable number of array-like arguments, each representing measurements
            from one condition. All arrays must have the same length (same subjects across
            conditions). Minimum 3 conditions required.
    
    Returns:
        TestResult with:
            - statistic: Friedman chi-squared statistic
            - pvalue: p-value from chi-squared distribution
            - df: None (df implicit in chi-squared approximation)
            - summary: DataFrame with test details
    
    Example:
        >>> from foodspec import run_friedman_test
        >>> condition1 = [10, 12, 11, 13, 15]
        >>> condition2 = [11, 13, 12, 14, 16]
        >>> condition3 = [12, 14, 13, 15, 17]
        >>> result = run_friedman_test(condition1, condition2, condition3)
        >>> print(f"χ²={result.statistic:.2f}, p={result.pvalue:.4f}")
    
    See Also:
        run_anova: Parametric repeated measures ANOVA
        run_wilcoxon_signed_rank: For exactly 2 paired conditions
        run_kruskal_wallis: For independent (unpaired) groups
    """
    stat, p = stats.friedmanchisquare(*samples)
    summary = pd.DataFrame([{"test": "friedman", "statistic": stat, "pvalue": p, "df": None}])
    return TestResult(statistic=float(stat), pvalue=float(p), df=None, summary=summary)


# Convenience alias expected by some users/tests
def tukey_hsd(values, groups, alpha: float = 0.05):
    return run_tukey_hsd(values, groups, alpha=alpha)
