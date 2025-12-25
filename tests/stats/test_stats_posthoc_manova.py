import numpy as np
import pandas as pd
import pytest

from foodspec.stats import games_howell, run_manova


def test_games_howell_detects_difference():
    rng = np.random.default_rng(0)
    g1 = rng.normal(0, 1.0, 15)
    g2 = rng.normal(4, 2.0, 20)
    g3 = rng.normal(0.5, 1.0, 12)
    values = np.concatenate([g1, g2, g3])
    groups = np.array(["A"] * len(g1) + ["B"] * len(g2) + ["C"] * len(g3))

    tbl = games_howell(values, groups)

    # at least one strong difference (A vs B) should be significant
    ab_row = tbl[(tbl["group1"] == "A") & (tbl["group2"] == "B")]
    assert not ab_row.empty
    assert ab_row["p_adj"].iloc[0] < 0.05


def test_run_manova_detects_group_difference():
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        pytest.skip("statsmodels not installed")

    rng = np.random.default_rng(1)
    g1 = rng.normal(0, 0.5, size=(20, 2))
    g2 = rng.normal(2, 0.5, size=(20, 2))
    data = pd.DataFrame(np.vstack([g1, g2]), columns=["f1", "f2"])
    groups = np.array(["A"] * len(g1) + ["B"] * len(g2))

    res = run_manova(data, groups)
    assert res.pvalue < 0.05
