"""Tests for model lifecycle management."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from foodspec.ml.lifecycle import (
    ModelAgingScore,
    ModelLifecycleTracker,
    ModelState,
    PerformanceSnapshot,
    SunsetRule,
)


def test_lifecycle_tracker_initialization():
    """Test lifecycle tracker creation."""
    deployment_date = datetime(2025, 1, 1)
    tracker = ModelLifecycleTracker(
        model_id="model_v1",
        deployment_date=deployment_date,
        baseline_performance=0.90,
        metric_name="accuracy",
    )

    assert tracker.model_id == "model_v1"
    assert tracker.deployment_date == deployment_date
    assert tracker.baseline_performance == 0.90
    assert tracker.state == ModelState.CHAMPION
    assert len(tracker.snapshots) == 0


def test_record_performance():
    """Test recording performance snapshots."""
    tracker = ModelLifecycleTracker(
        model_id="test_model",
        deployment_date=datetime(2025, 1, 1),
        baseline_performance=0.90,
    )

    tracker.record_performance(
        timestamp=datetime(2025, 1, 2),
        metric_value=0.88,
        n_samples=100,
        batch="batch_a",
    )

    tracker.record_performance(
        timestamp=datetime(2025, 1, 3),
        metric_value=0.87,
        n_samples=150,
        batch="batch_b",
    )

    assert len(tracker.snapshots) == 2
    assert tracker.snapshots[0].metric_value == 0.88
    assert tracker.snapshots[1].metric_value == 0.87
    assert tracker.snapshots[0].metadata["batch"] == "batch_a"


def test_aging_score_insufficient_data():
    """Test aging score with insufficient snapshots."""
    tracker = ModelLifecycleTracker(
        model_id="test",
        deployment_date=datetime(2025, 1, 1),
        baseline_performance=0.90,
    )

    aging = tracker.compute_aging_score(current_time=datetime(2025, 1, 10))

    assert aging.age_days == 9.0
    assert aging.performance_decay == 0.0
    assert "Insufficient data" in aging.recommendation


def test_aging_score_stable_performance():
    """Test aging score with stable performance."""
    deployment = datetime(2025, 1, 1)
    tracker = ModelLifecycleTracker(
        model_id="stable_model",
        deployment_date=deployment,
        baseline_performance=0.90,
    )

    # Simulate stable performance over 10 days
    for i in range(10):
        tracker.record_performance(
            timestamp=deployment + timedelta(days=i + 1),
            metric_value=0.89 + np.random.randn() * 0.005,  # Small noise
            n_samples=100,
        )

    aging = tracker.compute_aging_score(current_time=deployment + timedelta(days=11))

    assert aging.age_days == 10.0
    assert aging.performance_decay < 0.05  # Less than 5% decay
    assert "OK" in aging.recommendation or "stable" in aging.recommendation.lower()
    assert not aging.is_retired


def test_aging_score_performance_decay():
    """Test aging score with significant decay."""
    deployment = datetime(2025, 1, 1)
    tracker = ModelLifecycleTracker(
        model_id="decaying_model",
        deployment_date=deployment,
        baseline_performance=0.90,
    )

    # Simulate linear decay: 0.90 → 0.75 over 10 days
    for i in range(10):
        perf = 0.90 - 0.015 * i  # Decay rate ~0.015/day
        tracker.record_performance(
            timestamp=deployment + timedelta(days=i + 1),
            metric_value=perf,
            n_samples=100,
        )

    aging = tracker.compute_aging_score(current_time=deployment + timedelta(days=11))

    assert aging.age_days == 10.0
    assert aging.performance_decay > 0.1  # More than 10% decay
    assert aging.decay_rate > 0.01  # Positive decay rate
    assert "WARNING" in aging.recommendation or "CAUTION" in aging.recommendation


def test_sunset_rule_age_limit():
    """Test sunset based on age limit."""
    deployment = datetime(2025, 1, 1)
    sunset_rule = SunsetRule(
        max_age_days=30.0,
        min_samples_evaluated=100,
        grace_period_days=7.0,
    )

    tracker = ModelLifecycleTracker(
        model_id="old_model",
        deployment_date=deployment,
        baseline_performance=0.90,
        sunset_rule=sunset_rule,
    )

    # Record stable performance for 35 days
    for i in range(35):
        tracker.record_performance(
            timestamp=deployment + timedelta(days=i + 1),
            metric_value=0.89,
            n_samples=10,
        )

    aging = tracker.compute_aging_score(current_time=deployment + timedelta(days=35))

    assert aging.is_retired  # Should be retired due to age
    assert "SUNSET" in aging.recommendation


def test_sunset_rule_performance_floor():
    """Test sunset based on performance floor."""
    deployment = datetime(2025, 1, 1)
    sunset_rule = SunsetRule(
        min_performance=0.80,
        min_samples_evaluated=100,
        grace_period_days=7.0,
    )

    tracker = ModelLifecycleTracker(
        model_id="poor_model",
        deployment_date=deployment,
        baseline_performance=0.90,
        sunset_rule=sunset_rule,
    )

    # Degrade to below threshold after grace period
    for i in range(10):
        perf = 0.90 - 0.02 * i  # Drop to 0.72
        tracker.record_performance(
            timestamp=deployment + timedelta(days=i + 1),
            metric_value=perf,
            n_samples=20,
        )

    aging = tracker.compute_aging_score(current_time=deployment + timedelta(days=11))

    assert aging.is_retired  # Should be retired due to low performance


def test_sunset_rule_grace_period():
    """Test sunset grace period (no retirement during burn-in)."""
    deployment = datetime(2025, 1, 1)
    sunset_rule = SunsetRule(
        min_performance=0.80,
        grace_period_days=7.0,
        min_samples_evaluated=50,
    )

    tracker = ModelLifecycleTracker(
        model_id="new_model",
        deployment_date=deployment,
        baseline_performance=0.90,
        sunset_rule=sunset_rule,
    )

    # Poor performance during grace period
    for i in range(5):
        tracker.record_performance(
            timestamp=deployment + timedelta(days=i + 1),
            metric_value=0.75,  # Below threshold
            n_samples=20,
        )

    aging = tracker.compute_aging_score(current_time=deployment + timedelta(days=6))

    # Should NOT be retired (still in grace period)
    assert not aging.is_retired


def test_get_performance_history():
    """Test exporting performance history."""
    deployment = datetime(2025, 1, 1)
    tracker = ModelLifecycleTracker(
        model_id="test",
        deployment_date=deployment,
        baseline_performance=0.90,
    )

    tracker.record_performance(
        timestamp=datetime(2025, 1, 2), metric_value=0.88, n_samples=100, batch="A"
    )
    tracker.record_performance(
        timestamp=datetime(2025, 1, 3), metric_value=0.87, n_samples=150, batch="B"
    )

    history = tracker.get_performance_history()

    assert len(history) == 2
    assert "timestamp" in history.columns
    assert "metric_value" in history.columns
    assert "n_samples" in history.columns
    assert "batch" in history.columns
    assert history["metric_value"].tolist() == [0.88, 0.87]
    assert history["batch"].tolist() == ["A", "B"]


def test_model_state_enum():
    """Test ModelState enum values."""
    assert ModelState.DEVELOPMENT.value == "development"
    assert ModelState.CHAMPION.value == "champion"
    assert ModelState.CHALLENGER.value == "challenger"
    assert ModelState.RETIRED.value == "retired"


def test_sunset_rule_decay_rate():
    """Test sunset based on decay rate threshold."""
    deployment = datetime(2025, 1, 1)
    sunset_rule = SunsetRule(
        max_decay_rate=0.01,  # Max 0.01 perf units/day
        min_samples_evaluated=100,
        grace_period_days=7.0,
    )

    tracker = ModelLifecycleTracker(
        model_id="fast_decay",
        deployment_date=deployment,
        baseline_performance=0.90,
        sunset_rule=sunset_rule,
    )

    # Fast decay: 0.02/day (exceeds threshold)
    for i in range(10):
        perf = 0.90 - 0.02 * i
        tracker.record_performance(
            timestamp=deployment + timedelta(days=i + 1),
            metric_value=perf,
            n_samples=15,
        )

    aging = tracker.compute_aging_score(current_time=deployment + timedelta(days=11))

    assert aging.is_retired  # Should retire due to high decay rate
