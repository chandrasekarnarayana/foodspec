"""Tests for model calibration diagnostics."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from foodspec.ml.calibration import (
    CalibrationDiagnostics,
    calibration_slope_intercept,
    compute_calibration_diagnostics,
    recalibrate_classifier,
)


def test_calibration_diagnostics_perfect():
    """Test calibration metrics for perfectly calibrated predictions."""
    # Perfect calibration: predicted probabilities match true frequencies
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

    diag = compute_calibration_diagnostics(y_true, y_proba, n_bins=5)

    assert isinstance(diag, CalibrationDiagnostics)
    assert diag.slope == pytest.approx(1.0, abs=0.2)  # Close to ideal slope
    assert diag.intercept == pytest.approx(0.0, abs=0.3)  # Close to ideal intercept
    assert diag.ece < 0.15  # Low calibration error
    assert diag.brier_score < 0.3  # Low Brier score
    assert diag.n_bins == 5
    assert len(diag.reliability_curve) == 5


def test_calibration_diagnostics_overconfident():
    """Test calibration metrics for overconfident model."""
    # Overconfident: high probabilities for both classes
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.05, 0.1, 0.05, 0.9, 0.95, 0.9, 0.95, 0.99])

    diag = compute_calibration_diagnostics(y_true, y_proba, n_bins=5)

    # Overconfident models tend to have slope > 1
    assert diag.slope > 0.5  # Positive slope (not perfectly calibrated)
    assert diag.ece > 0.0  # Some calibration error


def test_calibration_diagnostics_quantile_binning():
    """Test quantile-based binning strategy."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 100)
    y_proba = np.random.beta(2, 2, 100)

    diag = compute_calibration_diagnostics(y_true, y_proba, n_bins=10, strategy="quantile")

    assert diag.n_bins == 10
    assert len(diag.reliability_curve) == 10
    assert all(diag.reliability_curve["count"] > 0)  # Quantile binning ensures non-empty bins


def test_calibration_slope_intercept():
    """Test calibration slope/intercept extraction."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.6, 0.7, 0.8])

    slope, intercept = calibration_slope_intercept(y_true, y_proba)

    assert isinstance(slope, float)
    assert isinstance(intercept, float)
    assert 0.5 < slope < 2.0  # Reasonable range


def test_recalibrate_classifier_platt():
    """Test Platt scaling recalibration."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train base classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    # Recalibrate
    calibrated_clf = recalibrate_classifier(clf, X_cal, y_cal, method="platt", cv=3)

    # Test prediction
    y_proba = calibrated_clf.predict_proba(X_cal)[:, 1]

    assert y_proba.shape == (len(y_cal),)
    assert np.all((y_proba >= 0) & (y_proba <= 1))


def test_recalibrate_classifier_isotonic():
    """Test isotonic regression recalibration."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5, random_state=42)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    calibrated_clf = recalibrate_classifier(clf, X_cal, y_cal, method="isotonic", cv=3)

    y_proba = calibrated_clf.predict_proba(X_cal)[:, 1]

    assert y_proba.shape == (len(y_cal),)
    assert np.all((y_proba >= 0) & (y_proba <= 1))


def test_is_well_calibrated():
    """Test calibration quality threshold checks."""
    diag_good = CalibrationDiagnostics(
        slope=1.0,
        intercept=0.0,
        bias=0.01,
        ece=0.03,
        mce=0.05,
        brier_score=0.15,
        n_bins=10,
        reliability_curve=None,
    )

    assert diag_good.is_well_calibrated(slope_tol=0.1, intercept_tol=0.05, ece_threshold=0.05)

    diag_bad = CalibrationDiagnostics(
        slope=1.5,
        intercept=0.2,
        bias=0.1,
        ece=0.15,
        mce=0.25,
        brier_score=0.30,
        n_bins=10,
        reliability_curve=None,
    )

    assert not diag_bad.is_well_calibrated(slope_tol=0.1, intercept_tol=0.05, ece_threshold=0.05)
