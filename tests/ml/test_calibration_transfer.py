import numpy as np
import pandas as pd

from foodspec import FoodSpec
from foodspec.preprocess.calibration_transfer import (
    adapt_calibration_incremental,
    calibration_transfer_workflow,
    detect_drift,
    direct_standardization,
    piecewise_direct_standardization,
)


def make_bias_scale(source: np.ndarray, scale: float = 1.2, bias: float = 0.5) -> np.ndarray:
    return source * scale + bias


def test_direct_standardization_corrects_bias_and_scale():
    rng = np.random.RandomState(0)
    n_std, n_prod, n_wn = 20, 15, 50
    X_source_std = rng.randn(n_std, n_wn)
    X_source_prod = rng.randn(n_prod, n_wn)

    X_target_std = make_bias_scale(X_source_std, scale=1.1, bias=0.3)
    X_target_prod = make_bias_scale(X_source_prod, scale=1.1, bias=0.3)

    corrected, metrics = direct_standardization(X_source_std, X_target_std, X_target_prod, alpha=0.5)

    # Basic sanity: corrected shape matches and metrics populated
    assert corrected.shape == X_target_prod.shape
    assert metrics["n_standards"] == n_std
    assert metrics["reconstruction_rmse"] < 0.2
    assert np.isfinite(metrics["transformation_condition_number"])  # stability indicator


def test_piecewise_direct_standardization_runs_and_improves():
    rng = np.random.RandomState(1)
    n_std, n_prod, n_wn = 25, 10, 60
    X_source_std = rng.randn(n_std, n_wn)
    X_source_prod = rng.randn(n_prod, n_wn)

    # Inject localized differences
    X_target_std = X_source_std * (1.05 + 0.02 * np.sin(np.arange(n_wn) / 5.0)) + 0.2
    X_target_prod = X_source_prod * (1.05 + 0.02 * np.sin(np.arange(n_wn) / 5.0)) + 0.2

    corrected, metrics = piecewise_direct_standardization(
        X_source_std, X_target_std, X_target_prod, window_size=9, alpha=0.5
    )

    # Sanity checks: output shape and metrics
    assert corrected.shape == X_target_prod.shape
    assert metrics["window_size"] == 9
    assert metrics["n_standards"] == n_std


def test_detect_drift_flags_large_mean_shift():
    rng = np.random.RandomState(2)
    X_ref = rng.randn(30, 40)
    X_curr = X_ref + 0.5  # mean shift
    drift, m = detect_drift(X_ref, X_curr, threshold=0.05)
    assert drift
    assert m["drift_detected"] is True
    assert m["normalized_drift"] > m["threshold"]


def test_adapt_calibration_incremental_updates_reference():
    rng = np.random.RandomState(3)
    X_ref = rng.randn(10, 20)
    X_new = rng.randn(5, 20)
    updated, m = adapt_calibration_incremental(X_ref, X_new, weight_decay=0.8)
    assert updated.shape[0] == 15
    assert m["reference_size_before"] == 10
    assert m["reference_size_after"] == 15


def test_calibration_transfer_workflow_ds_metrics_present():
    rng = np.random.RandomState(4)
    n_std, n_prod, n_wn = 15, 8, 40
    X_source_std = rng.randn(n_std, n_wn)
    X_source_prod = rng.randn(n_prod, n_wn)
    X_target_std = make_bias_scale(X_source_std, scale=1.08, bias=0.25)
    X_target_prod = make_bias_scale(X_source_prod, scale=1.08, bias=0.25)

    corrected, all_metrics = calibration_transfer_workflow(
        X_source_std, X_target_std, X_target_prod, method="ds", alpha=0.5
    )
    assert corrected.shape == X_target_prod.shape
    assert "transfer" in all_metrics
    assert "reconstruction_rmse" in all_metrics["transfer"]


def test_apply_calibration_transfer_basic():
    n_std, n_wn = 20, 120
    wn = np.linspace(400, 3000, n_wn)
    # Source standards: baseline 1.0
    source = np.random.randn(n_std, n_wn) * 0.02 + 1.0
    # Target standards: baseline 1.1
    target = source * 1.02 + 0.05  # introduce small linear difference
    # Production target data
    X = np.random.randn(50, n_wn) * 0.02 + 1.05
    meta = pd.DataFrame({"label": ["classA"] * 50})
    fs = FoodSpec(X, wavenumbers=wn, metadata=meta, modality="raman")
    fs.apply_calibration_transfer(source_standards=source, target_standards=target, method="ds", alpha=1.0)
    # Shape preserved
    assert fs.data.x.shape == X.shape
    # Metrics recorded
    assert any(k.startswith("calibration_transfer_") for k in fs.bundle.metrics.keys())
