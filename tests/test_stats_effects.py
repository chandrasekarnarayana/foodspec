import numpy as np

from foodspec.stats.effects import compute_cohens_d, compute_anova_effect_sizes


def test_compute_cohens_d_pooled_and_unpooled():
    g1 = [1, 2, 3, 4]
    g2 = [2, 3, 4, 5]
    d_pooled = compute_cohens_d(g1, g2, pooled=True)
    d_unpooled = compute_cohens_d(g1, g2, pooled=False)
    assert d_pooled < 0  # mean g1 < g2
    assert d_unpooled < 0


def test_compute_anova_effect_sizes():
    res = compute_anova_effect_sizes(ss_between=10, ss_total=20, ss_within=5)
    assert np.isclose(res["eta_squared"], 0.5)
    assert np.isclose(res["partial_eta_squared"], 10 / 15)

    res2 = compute_anova_effect_sizes(ss_between=0, ss_total=0, ss_within=None)
    assert np.isnan(res2["eta_squared"])

    # partial with zero denom
    res3 = compute_anova_effect_sizes(ss_between=1, ss_total=2, ss_within=0)
    assert "partial_eta_squared" in res3
