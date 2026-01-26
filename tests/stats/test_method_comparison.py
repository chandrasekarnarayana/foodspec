from __future__ import annotations

import numpy as np

from foodspec.stats.method_comparison import lins_concordance_correlation, passing_bablok


def test_passing_bablok_identity() -> None:
    x = np.arange(10, dtype=float)
    y = x + 0.01
    res = passing_bablok(x, y)
    assert abs(res.slope - 1.0) < 0.1


def test_lins_ccc() -> None:
    x = np.arange(10, dtype=float)
    y = x + 0.01
    ccc = lins_concordance_correlation(x, y)
    assert ccc > 0.95
