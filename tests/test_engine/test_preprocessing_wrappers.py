import numpy as np

from foodspec.engine.preprocessing import baseline_als, normalize_vector, smooth_savgol


def test_engine_preprocessing_wrappers():
    X = np.random.rand(3, 50)
    baseline = baseline_als(X)
    assert baseline.shape == X.shape

    smoothed = smooth_savgol(X, window_length=7, polyorder=3)
    assert smoothed.shape == X.shape

    normed = normalize_vector(X, norm="l2")
    assert normed.shape == X.shape

