from __future__ import annotations

import numpy as np
import pandas as pd

from foodspec.stats.hypothesis_tests import (
    check_group_sequential,
    group_sequential_boundaries,
    run_ancova,
    run_anderson_darling,
    run_bartlett,
    run_levene,
    run_noninferiority,
    run_tost_equivalence,
)


def test_anderson_darling_normality() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=50)
    res = run_anderson_darling(data)
    assert np.isfinite(res.statistic)
    assert 0 <= res.pvalue <= 1


def test_homoscedasticity_tests() -> None:
    g1 = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    g2 = np.array([1.0, 1.02, 0.98, 1.01, 0.99])
    res_l = run_levene(g1, g2)
    res_b = run_bartlett(g1, g2)
    assert 0 <= res_l.pvalue <= 1
    assert 0 <= res_b.pvalue <= 1


def test_equivalence_and_noninferiority() -> None:
    g1 = np.array([1.0, 1.05, 0.95, 1.1, 0.9])
    g2 = np.array([1.0, 1.0, 0.98, 1.02, 1.01])
    eq = run_tost_equivalence(g1, g2, delta=0.5)
    assert eq.equivalent
    ni = run_noninferiority(g1, g2, margin=0.5)
    assert ni.noninferior


def test_group_sequential_boundaries() -> None:
    bounds = group_sequential_boundaries(n_looks=3, alpha=0.05)
    assert bounds.size == 3
    res = check_group_sequential([0.5, 4.0, 0.2], bounds)
    assert res["crossed"]


def test_ancova_basic() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            "group": ["A", "A", "A", "B", "B", "B"],
            "cov": [0.2, 0.3, 0.1, 0.4, 0.5, 0.35],
        }
    )
    res = run_ancova(df, dv="y", group="group", covariates=["cov"])
    assert np.isfinite(res.statistic)
    assert 0 <= res.pvalue <= 1
