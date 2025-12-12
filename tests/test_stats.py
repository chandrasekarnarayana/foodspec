import numpy as np
import pandas as pd

from foodspec.stats import (
    bootstrap_metric,
    check_minimum_samples,
    compute_cohens_d,
    compute_correlation_matrix,
    compute_correlations,
    compute_cross_correlation,
    permutation_test_metric,
    run_anova,
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_ttest,
    run_tukey_hsd,
    run_wilcoxon_signed_rank,
)


def test_ttest_two_sample_detects_difference():
    g1 = np.random.default_rng(0).normal(0, 1, size=30)
    g2 = np.random.default_rng(1).normal(2, 1, size=30)
    res = run_ttest(g1, g2)
    assert res.pvalue < 0.001


def test_anova_detects_group_difference():
    data = np.array([1, 1.1, 1.2, 5.0, 5.1, 5.2])
    groups = ["a", "a", "a", "b", "b", "b"]
    res = run_anova(data, groups)
    assert res.pvalue < 0.001


def test_tukey_hsd_runs_if_statsmodels_present():
    try:
        tbl = run_tukey_hsd([1, 1.1, 1.2, 5.0, 5.1, 5.2], ["a", "a", "a", "b", "b", "b"])
        assert "p_adj" in tbl.columns
    except ImportError:
        # statsmodels optional; acceptable to skip
        pass


def test_correlations_and_matrix():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    res = compute_correlations(df, ("x", "y"), method="pearson")
    assert np.isclose(res["r"], 1.0)
    mat = compute_correlation_matrix(df, ["x", "y"], method="pearson")
    assert mat.loc["x", "y"] > 0.99


def test_cross_correlation():
    seq1 = [1, 2, 3, 4]
    seq2 = [1, 2, 3, 4]
    df = compute_cross_correlation(seq1, seq2, max_lag=1)
    zero_corr = df.loc[df["lag"] == 0, "correlation"].iloc[0]
    assert zero_corr > 0


def test_cohens_d_sign():
    g1 = [1, 1, 1]
    g2 = [2, 2, 2]
    d = compute_cohens_d(g1, g2)
    assert d < 0  # group1 mean < group2 mean


def test_check_minimum_samples():
    groups = ["a", "a", "b"]
    df = check_minimum_samples(groups, min_per_group=2)
    assert bool(df.loc[df["group"] == "a", "ok"].iloc[0]) is True
    assert bool(df.loc[df["group"] == "b", "ok"].iloc[0]) is False


def test_nonparametric_tests_detect_difference():
    df = pd.DataFrame(
        {
            "value": [1, 1.1, 1.2, 1.05, 1.15, 3.0, 3.1, 3.2, 3.05, 3.15],
            "group": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        }
    )
    res_kw = run_kruskal_wallis(df, group_col="group", value_col="value")
    assert res_kw.pvalue < 0.02
    res_u = run_mannwhitney_u(df, group_col="group", value_col="value")
    assert res_u.pvalue < 0.01
    res_w = run_wilcoxon_signed_rank(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
    )
    assert res_w.pvalue < 0.05


def test_bootstrap_and_permutation_metric():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0])

    def acc(a, b):
        return np.mean(a == b)

    boot = bootstrap_metric(acc, y_true, y_pred, n_bootstrap=100, random_state=0)
    assert "bootstrap_samples" in boot and len(boot["bootstrap_samples"]) == 100
    perm = permutation_test_metric(acc, y_true, y_pred, n_permutations=100, random_state=0)
    assert 0 <= perm["p_value"] <= 1
