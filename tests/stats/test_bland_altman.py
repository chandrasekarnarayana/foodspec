import numpy as np

from foodspec.stats.method_comparison import bland_altman, bland_altman_plot


def test_bland_altman_basic():
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    b = np.array([1.1, 2.1, 2.9, 4.2, 4.8], dtype=float)
    res = bland_altman(a, b)
    assert isinstance(res.mean_diff, float)
    assert res.loa_high > res.loa_low
    fig = bland_altman_plot(a, b)
    assert fig is not None
