import numpy as np
import pandas as pd

from foodspec.features.selection import stability_selection


def test_stability_selection_reproducible():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 5)), columns=[f"f{i}" for i in range(5)])
    labels = np.array([0, 1] * 25)

    res1 = stability_selection(df, labels, n_resamples=10, sample_fraction=0.8, seed=123)
    res2 = stability_selection(df, labels, n_resamples=10, sample_fraction=0.8, seed=123)

    assert np.allclose(res1["frequency"], res2["frequency"])
    assert (res1["frequency"] >= 0).all()
    assert (res1["frequency"] <= 1).all()
