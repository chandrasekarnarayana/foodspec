import numpy as np

from foodspec.calibration_transfer import compute_transfer_success_metrics


def test_compute_transfer_success_metrics_improvement_and_leverage():
    rng = np.random.RandomState(11)
    n_val = 50
    n_prod = 40
    y_true = rng.randn(n_val)

    # Source model close to truth
    y_pred_source = y_true + rng.randn(n_val) * 0.05
    # Target before transfer worse
    y_pred_target_before = y_true + rng.randn(n_val) * 0.30 + 0.5
    # Target after transfer better (simulate improvement)
    y_pred_target_after = y_true + rng.randn(n_val) * 0.10 + 0.1

    # Production spectra: include a few outliers to trigger high leverage
    X_target_prod = rng.randn(n_prod, 60)
    X_target_prod[:3] += 5.0  # extreme values for leverage

    metrics = compute_transfer_success_metrics(
        y_true,
        y_pred_source,
        y_pred_target_before,
        y_pred_target_after,
        X_target_prod,
        leverage_threshold=0.5,
    )

    # RMSE and R2 improvements
    assert metrics["target_after_transfer"]["rmse"] < metrics["target_before_transfer"]["rmse"]
    assert metrics["improvement"]["rmse_improvement_ratio"] > 0
    assert metrics["target_after_transfer"]["r2"] >= metrics["target_before_transfer"]["r2"]

    # Leverage outliers
    lev = metrics["leverage_outliers"]
    assert lev["high_leverage_count"] >= 1
    assert 0.0 < lev["leverage_rate"] <= 1.0

    # Residual stats present
    res = metrics["residuals"]
    assert np.isfinite(res["mean"]) and np.isfinite(res["std"])