from __future__ import annotations

import numpy as np

from foodspec.stats.doe import (
    central_composite_design,
    d_optimal_design,
    fractional_factorial_2level,
    full_factorial_2level,
    randomized_block_design,
)


def test_full_factorial() -> None:
    design = full_factorial_2level(["A", "B", "C"])
    assert design.shape[0] == 8


def test_fractional_factorial() -> None:
    design = fractional_factorial_2level(["A", "B", "C"], {"D": "AB"})
    assert "D" in design.columns
    assert set(design["D"].unique()) <= {-1, 1}


def test_central_composite() -> None:
    design = central_composite_design(["A", "B"])
    assert design.shape[1] == 2
    assert np.allclose(design.iloc[-1].to_numpy(), 0.0)


def test_randomized_block() -> None:
    design = randomized_block_design(["t1", "t2"], ["b1", "b2"], random_state=0)
    assert design.shape[0] == 4


def test_d_optimal() -> None:
    candidates = full_factorial_2level(["A", "B"]).to_numpy()
    res = d_optimal_design(candidates, n_runs=3, random_state=0)
    assert res.design.shape[0] == 3
