import numpy as np
import pandas as pd
import pytest

from foodspec.stats.hypothesis_tests import pairwise_tukeyhsd, run_manova, tukey_hsd


def test_run_manova_and_tukey():
    # Synthetic small dataset
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": np.repeat(["A", "B", "C"], 10),
        "f1": np.concatenate([rng.normal(0, 1, 10), rng.normal(1, 1, 10), rng.normal(2, 1, 10)]),
        "f2": np.concatenate([rng.normal(0, 1, 10), rng.normal(1, 1, 10), rng.normal(2, 1, 10)]),
    })
    res = run_manova(df, group_col="group", dependent_cols=["f1", "f2"])
    assert "group" in str(res)

    # Tukey on a single feature (skip if statsmodels not installed)
    if pairwise_tukeyhsd is None:
        pytest.skip("statsmodels not available for Tukey HSD")
    tuk = tukey_hsd(df["f1"].to_numpy(), df["group"].to_numpy())
    assert "reject" in tuk.columns
