from __future__ import annotations

import numpy as np

from foodspec.stats.distribution_fitting import compare_distributions, fit_distribution, probability_plot_data


def test_fit_distribution_gamma() -> None:
    rng = np.random.default_rng(0)
    data = rng.gamma(shape=2.0, scale=1.0, size=80)
    res = fit_distribution(data, "gamma")
    assert res.ks_pvalue >= 0.0
    assert res.aic == res.aic


def test_compare_distributions() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=50)
    fits = compare_distributions(data, ["normal", "lognorm"])
    assert len(fits) == 2


def test_probability_plot_data() -> None:
    rng = np.random.default_rng(2)
    data = rng.normal(size=40)
    info = probability_plot_data(data, dist_name="normal")
    assert info["theoretical"].shape[0] == info["ordered"].shape[0]
