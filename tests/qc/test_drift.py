"""Tests for production drift detection."""

import numpy as np
import pandas as pd
import pytest

from foodspec.qc.drift import (
    DriftReport,
    detect_drift,
    detect_feature_drift,
    kl_divergence,
    population_stability_index,
    should_recalibrate,
)


def test_psi_no_drift():
    """Test PSI for identical distributions (no drift)."""
    reference = np.random.randn(1000)
    current = reference + np.random.randn(1000) * 0.01  # Tiny noise

    psi = population_stability_index(reference, current, bins=10)

    assert psi < 0.1  # No significant drift


def test_psi_moderate_drift():
    """Test PSI for moderate distributional shift."""
    np.random.seed(42)
    reference = np.random.randn(1000)
    current = reference + 0.5  # Shift mean by 0.5 std

    psi = population_stability_index(reference, current, bins=10)

    assert 0.1 < psi < 0.5  # Moderate to significant drift


def test_psi_significant_drift():
    """Test PSI for large distributional shift."""
    np.random.seed(42)
    reference = np.random.randn(1000)
    current = np.random.randn(1000) + 2.0  # Large shift

    psi = population_stability_index(reference, current, bins=10)

    assert psi > 0.25  # Significant drift


def test_kl_divergence_identical():
    """Test KL divergence for identical distributions."""
    reference = np.random.randn(1000)
    current = reference.copy()

    kl = kl_divergence(reference, current, bins=10)

    assert kl < 0.01  # Near zero for identical distributions


def test_kl_divergence_different():
    """Test KL divergence for different distributions."""
    np.random.seed(42)
    reference = np.random.randn(1000)
    current = np.random.randn(1000) * 2 + 1  # Different scale and location

    kl = kl_divergence(reference, current, bins=10)

    assert kl > 0.1  # Significant KL divergence


def test_detect_drift_stable():
    """Test drift detection for stable distribution."""
    np.random.seed(42)
    reference = np.random.randn(500)
    current = np.random.randn(500)  # Same distribution

    report = detect_drift(reference, current, psi_threshold=0.25, ks_alpha=0.05)

    assert isinstance(report, DriftReport)
    assert not report.drift_detected  # Should not detect drift
    assert report.psi < 0.25
    assert "OK" in report.recommendation or "ok" in report.recommendation.lower()


def test_detect_drift_significant():
    """Test drift detection for significant shift."""
    np.random.seed(42)
    reference = np.random.randn(500)
    current = np.random.randn(500) + 1.5  # Significant shift

    report = detect_drift(reference, current, psi_threshold=0.25, ks_alpha=0.05)

    assert report.drift_detected  # Should detect drift
    assert report.psi > 0.1  # Some PSI detected
    assert "WARNING" in report.recommendation or "CRITICAL" in report.recommendation


def test_detect_drift_continuous():
    """Test drift detection with explicit continuous method."""
    np.random.seed(42)
    reference = np.random.uniform(0, 1, 300)
    current = np.random.uniform(0.3, 1.3, 300)  # Shifted range

    report = detect_drift(reference, current, method="continuous")

    assert isinstance(report.wasserstein_distance, float)
    assert report.wasserstein_distance > 0
    assert report.ks_pvalue < 1.0


def test_detect_feature_drift():
    """Test feature-level drift detection."""
    np.random.seed(42)
    reference = pd.DataFrame({
        "feat1": np.random.randn(200),
        "feat2": np.random.randn(200),
        "feat3": np.random.randn(200),
    })

    current = pd.DataFrame({
        "feat1": np.random.randn(200),  # No drift
        "feat2": np.random.randn(200) + 2.0,  # Significant drift
        "feat3": np.random.randn(200) * 2,  # Scale drift
    })

    reports = detect_feature_drift(reference, current, psi_threshold=0.25)

    assert "feat1" in reports
    assert "feat2" in reports
    assert "feat3" in reports

    # feat1 should be stable
    assert reports["feat1"].psi < 0.25

    # feat2 should have significant drift
    assert reports["feat2"].psi > 0.25


def test_detect_feature_drift_subset():
    """Test drift detection on specific features only."""
    np.random.seed(42)
    reference = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "feat3": np.random.randn(100),
    })

    current = pd.DataFrame({
        "feat1": np.random.randn(100) + 1,
        "feat2": np.random.randn(100),
        "feat3": np.random.randn(100),
    })

    reports = detect_feature_drift(reference, current, features=["feat1", "feat2"])

    assert "feat1" in reports
    assert "feat2" in reports
    assert "feat3" not in reports  # Not analyzed


def test_should_recalibrate_performance_drop():
    """Test recalibration trigger based on performance drop."""
    reference_perf = 0.90
    current_perf = 0.84  # 6% drop

    should_recal = should_recalibrate(
        reference_perf, current_perf, performance_threshold=0.05
    )

    assert should_recal  # Should trigger recalibration


def test_should_recalibrate_drift():
    """Test recalibration trigger based on drift."""
    reference_perf = 0.90
    current_perf = 0.89  # Small drop (acceptable)

    drift_report = DriftReport(
        drift_detected=True,
        psi=0.30,  # Significant drift
        kl_divergence=0.5,
        ks_statistic=0.3,
        ks_pvalue=0.001,
        wasserstein_distance=0.4,
        recommendation="CRITICAL",
    )

    should_recal = should_recalibrate(
        reference_perf,
        current_perf,
        performance_threshold=0.05,
        drift_report=drift_report,
        psi_threshold=0.25,
    )

    assert should_recal  # Drift should trigger recalibration


def test_should_recalibrate_stable():
    """Test no recalibration needed when stable."""
    reference_perf = 0.90
    current_perf = 0.89  # Small acceptable drop

    drift_report = DriftReport(
        drift_detected=False,
        psi=0.05,  # No significant drift
        kl_divergence=0.01,
        ks_statistic=0.1,
        ks_pvalue=0.5,
        wasserstein_distance=0.1,
        recommendation="OK",
    )

    should_recal = should_recalibrate(
        reference_perf,
        current_perf,
        performance_threshold=0.05,
        drift_report=drift_report,
        psi_threshold=0.25,
    )

    assert not should_recal  # Stable, no recalibration needed
