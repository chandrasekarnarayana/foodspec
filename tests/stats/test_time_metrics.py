import numpy as np
from foodspec.stats.time_metrics import linear_slope, quadratic_acceleration, rolling_slope


def test_linear_slope_and_acceleration():
    t = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = 0.5 * t + 1.0
    m, b = linear_slope(t, y)
    assert np.isclose(m, 0.5, atol=1e-8)
    assert np.isclose(b, 1.0, atol=1e-8)
    acc = quadratic_acceleration(t, y)
    assert np.isclose(acc, 0.0, atol=1e-8)


def test_rolling_slope_basic():
    t = np.arange(10, dtype=float)
    y = 2.0 * t
    rs = rolling_slope(t, y, window=3)
    # After window-1, slopes should be ~2.0
    assert np.allclose(rs[2:], 2.0)
