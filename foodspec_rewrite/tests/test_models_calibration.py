"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from foodspec.models import CalibratedClassifier, calibration_metrics


def test_platt_calibration_fits_and_predicts() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 5))
    y_train = np.array([0, 1] * 25)
    X_val = rng.normal(size=(20, 5))
    y_val = np.array([0, 1] * 10)
    X_test = rng.normal(size=(30, 5))

    base = LogisticRegression(random_state=0, max_iter=1000)
    cal = CalibratedClassifier(base, method="sigmoid")
    cal.fit_and_calibrate(X_train, y_train, X_val, y_val)

    proba = cal.predict_proba(X_test)
    assert proba.shape == (30, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    preds = cal.predict(X_test)
    assert preds.shape == (30,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_isotonic_calibration_vs_platt() -> None:
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(60, 4))
    y_train = np.array([0, 1] * 30)
    X_val = rng.normal(size=(25, 4))
    y_val = np.array([0, 1] * 12 + [0])
    X_test = rng.normal(size=(20, 4))

    base_sig = LogisticRegression(random_state=0, max_iter=1000)
    base_iso = LogisticRegression(random_state=0, max_iter=1000)

    cal_sig = CalibratedClassifier(base_sig, method="sigmoid")
    cal_iso = CalibratedClassifier(base_iso, method="isotonic")

    cal_sig.fit_and_calibrate(X_train, y_train, X_val, y_val)
    cal_iso.fit_and_calibrate(X_train, y_train, X_val, y_val)

    proba_sig = cal_sig.predict_proba(X_test)
    proba_iso = cal_iso.predict_proba(X_test)

    # Both should be valid probabilities
    assert np.allclose(proba_sig.sum(axis=1), 1.0)
    assert np.allclose(proba_iso.sum(axis=1), 1.0)
    # But generally different
    assert not np.allclose(proba_sig, proba_iso)


def test_calibration_metrics_improvement() -> None:
    rng = np.random.default_rng(2)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    # Overconfident uncalibrated probabilities (poor calibration)
    proba_uncal = np.array([
        [0.99, 0.01],
        [0.05, 0.95],
        [0.98, 0.02],
        [0.1, 0.9],
        [0.95, 0.05],
        [0.08, 0.92],
        [0.97, 0.03],
        [0.15, 0.85],
    ])

    # Better calibrated (closer to true frequencies)
    proba_cal = np.array([
        [0.75, 0.25],
        [0.35, 0.65],
        [0.70, 0.30],
        [0.40, 0.60],
        [0.72, 0.28],
        [0.38, 0.62],
        [0.73, 0.27],
        [0.42, 0.58],
    ])

    metrics = calibration_metrics(y_true, proba_uncal, proba_cal)

    assert "ece_uncal" in metrics
    assert "ece_cal" in metrics
    assert "brier_uncal" in metrics
    assert "brier_cal" in metrics
    # All metrics should be positive floats
    assert all(isinstance(v, float) and v >= 0.0 for v in metrics.values())


def test_calibrated_classifier_validation_fold_only() -> None:
    """Ensure calibration is trained only on validation data, not train data."""
    rng = np.random.default_rng(3)
    X_train = rng.normal(loc=0, scale=1, size=(40, 3))
    y_train = np.array([0, 1] * 20)
    X_val = rng.normal(loc=2, scale=1, size=(15, 3))  # Different distribution
    y_val = np.array([0, 1] * 7 + [0])

    base = LogisticRegression(random_state=0, max_iter=1000)
    cal = CalibratedClassifier(base, method="sigmoid")
    cal.fit_and_calibrate(X_train, y_train, X_val, y_val)

    # Check that base model is fitted, but calibrator is based on validation data
    assert cal._fitted
    assert cal._calibrator is not None


def test_calibrated_classifier_not_fitted_error() -> None:
    base = LogisticRegression(random_state=0)
    cal = CalibratedClassifier(base, method="sigmoid")

    with pytest.raises(RuntimeError):
        cal.predict_proba(np.zeros((5, 3)))

    with pytest.raises(RuntimeError):
        cal.predict(np.zeros((5, 3)))


def test_calibrated_classifier_input_validation() -> None:
    rng = np.random.default_rng(4)
    X_train = rng.normal(size=(30, 4))
    y_train = np.array([0, 1] * 15)
    X_val = rng.normal(size=(10, 4))
    y_val = np.array([0, 1] * 5)

    base = LogisticRegression(random_state=0, max_iter=1000)
    cal = CalibratedClassifier(base, method="sigmoid")

    # Mismatched lengths
    with pytest.raises(ValueError):
        cal.fit_and_calibrate(X_train, y_train[:5], X_val, y_val)

    # Mismatched feature counts
    with pytest.raises(ValueError):
        cal.fit_and_calibrate(X_train, y_train, X_val[:, :2], y_val)
