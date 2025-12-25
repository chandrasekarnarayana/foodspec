import numpy as np

from foodspec.preprocess.calibration_transfer import piecewise_direct_standardization


def test_pds_handles_large_window_and_small_window():
    rng = np.random.RandomState(123)
    X_src = rng.randn(6, 12)
    # target is source with mild noise and small bias
    X_tgt = X_src + 0.05 * rng.randn(6, 12) + 0.02

    # Window larger than feature count should not crash
    corrected_big, metrics_big = piecewise_direct_standardization(X_src, X_tgt, X_tgt, window_size=50)
    assert corrected_big.shape == X_src.shape
    assert isinstance(metrics_big, dict)
    # Basic metric sanity
    assert np.isfinite(metrics_big.get("reconstruction_rmse", np.nan))
    assert metrics_big.get("n_standards", 0) >= 1

    # Small window (edge case)
    corrected_small, metrics_small = piecewise_direct_standardization(X_src, X_tgt, X_tgt, window_size=1)
    assert corrected_small.shape == X_src.shape
    assert isinstance(metrics_small, dict)
    assert np.isfinite(metrics_small.get("reconstruction_rmse", np.nan))
    assert metrics_small.get("n_standards", 0) >= 1
