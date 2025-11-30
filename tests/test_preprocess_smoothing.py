import numpy as np

from foodspec.preprocess.smoothing import MovingAverageSmoother, SavitzkyGolaySmoother


def test_savgol_smoothing_simple_sequence():
    X = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=float)
    smoother = SavitzkyGolaySmoother(window_length=5, polyorder=2)
    out = smoother.transform(X)
    assert out.shape == X.shape
    assert np.all(np.isfinite(out))


def test_moving_average_smoothing():
    X = np.array([[1, 3, 5, 7, 9]], dtype=float)
    ma = MovingAverageSmoother(window_size=3)
    out = ma.transform(X)
    assert out.shape == X.shape
    assert np.allclose(out[0, 2], np.mean([5, 7, 9]))

