from __future__ import annotations

import numpy as np

from foodspec.stats.diagnostics import (
    adjusted_r2,
    aic_from_rss,
    bic_from_rss,
    cronbach_alpha,
    normal_tolerance_interval,
    runs_test,
)


def test_adjusted_r2() -> None:
    adj = adjusted_r2(0.9, n_samples=100, n_features=3)
    assert adj < 0.9


def test_aic_bic() -> None:
    aic = aic_from_rss(10.0, n_samples=20, n_params=3)
    bic = bic_from_rss(10.0, n_samples=20, n_params=3)
    assert aic != bic


def test_cronbach_alpha() -> None:
    data = np.tile(np.arange(10, dtype=float)[:, None], (1, 3))
    alpha = cronbach_alpha(data)
    assert alpha > 0.9


def test_runs_test() -> None:
    values = [1, 2, 1, 2, 1, 2, 1]
    res = runs_test(values)
    assert 0 <= res.pvalue <= 1


def test_tolerance_interval() -> None:
    values = np.linspace(0, 1, 20)
    lo, hi = normal_tolerance_interval(values, coverage=0.9, confidence=0.9)
    assert lo < hi
