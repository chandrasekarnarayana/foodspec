from __future__ import annotations

import numpy as np

from foodspec.qc.control_charts import (
    cusum_chart,
    ewma_chart,
    individuals_mr_chart,
    levey_jennings,
    xbar_r_chart,
)


def test_xbar_r_chart() -> None:
    data = np.arange(40, dtype=float)
    res = xbar_r_chart(data, subgroup_size=5)
    assert res.xbar.points.shape[0] == 8
    assert res.variability.points.shape[0] == 8


def test_individuals_mr_chart() -> None:
    data = np.linspace(0, 1, 20)
    res = individuals_mr_chart(data)
    assert res.xbar.points.shape[0] == 20
    assert res.variability.points.shape[0] == 19


def test_cusum_chart() -> None:
    data = np.random.default_rng(0).normal(size=30)
    res = cusum_chart(data)
    assert len(res["pos"]) == 30


def test_ewma_chart() -> None:
    data = np.random.default_rng(1).normal(size=30)
    res = ewma_chart(data)
    assert res["ewma"].shape[0] == 30


def test_levey_jennings() -> None:
    data = np.random.default_rng(2).normal(size=20)
    res = levey_jennings(data)
    assert res.points.shape[0] == 20
