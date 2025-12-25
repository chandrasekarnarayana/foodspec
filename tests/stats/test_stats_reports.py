import numpy as np
import pandas as pd

from foodspec.stats.effects import compute_cohens_d
from foodspec.stats.hypothesis_tests import benjamini_hochberg, run_shapiro
from foodspec.stats.reporting import stats_report_for_feature, stats_report_for_features_table


def test_shapiro_and_bh_correction():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=50)
    res = run_shapiro(vals)
    assert 0 <= res.pvalue <= 1
    pvals = [0.01, 0.02, 0.2, 0.5]
    bh = benjamini_hochberg(pvals, alpha=0.05)
    assert "p_adj" in bh


def test_stats_report_per_feature_two_groups():
    rng = np.random.default_rng(1)
    g1 = rng.normal(0.0, 1.0, size=30)
    g2 = rng.normal(0.5, 1.0, size=30)
    vals = np.concatenate([g1, g2])
    groups = np.array(["a"] * len(g1) + ["b"] * len(g2))
    report = stats_report_for_feature(vals, groups, alpha=0.1, feature_name="feat")
    assert "p_adj" in report.columns
    assert report.shape[0] == 1
    d = compute_cohens_d(g1, g2)
    assert np.isfinite(d)


def test_stats_report_table_multi_feature():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "group": np.repeat(["x", "y", "z"], 10),
        "f1": rng.normal(0, 1, size=30),
        "f2": rng.normal(1, 1, size=30),
    })
    report = stats_report_for_features_table(df, group_col="group", alpha=0.1)
    assert set(report["feature"]) == {"f1", "f2"}
    assert "p_adj" in report.columns
