import numpy as np
import pandas as pd
import pytest

from foodspec.stats import (
    run_anova,
    run_ttest,
    run_manova,
    run_mannwhitney_u,
    run_wilcoxon_signed_rank,
    run_kruskal_wallis,
    run_friedman_test,
)


def test_ttest_variants():
    s1 = [1, 2, 3, 4]
    s2 = [2, 3, 4, 5]
    one_sample = run_ttest(s1, popmean=2.5)
    assert "pvalue" in one_sample.summary.columns or "pvalue" in one_sample.summary
    paired = run_ttest(s1, s2, paired=True)
    assert paired.df == len(s1) - 1
    two_sample = run_ttest(s1, s2, paired=False)
    assert two_sample.df == len(s1) + len(s2) - 2
    with pytest.raises(ValueError):
        run_ttest(s1)


def test_anova_and_nonparametric():
    data = [1, 2, 3, 4, 5, 6]
    groups = ["a", "a", "b", "b", "c", "c"]
    res = run_anova(data, groups)
    assert res.pvalue <= 1

    # Kruskal with group/value cols
    df = pd.DataFrame({"val": data, "grp": groups})
    kw = run_kruskal_wallis(df, group_col="grp", value_col="val")
    assert kw.pvalue <= 1

    mw = run_mannwhitney_u([1, 2, 3], [4, 5, 6], alternative="two-sided")
    assert mw.statistic >= 0


def test_paired_nonparametric_and_friedman():
    before = [1, 2, 3, 4]
    after = [2, 3, 4, 5]
    wilcox = run_wilcoxon_signed_rank(before, after)
    assert wilcox.pvalue <= 1

    f1 = [1, 2, 1]
    f2 = [2, 3, 2]
    f3 = [3, 4, 3]
    fr = run_friedman_test(f1, f2, f3)
    assert fr.pvalue <= 1


def test_manova_optional():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4]})
    groups = ["a", "a", "b"]
    try:
        res = run_manova(df, groups)
        assert res.pvalue <= 1
    except ImportError:
        pytest.skip("statsmodels not installed")
