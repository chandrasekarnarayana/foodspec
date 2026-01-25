"""
Unit tests for reliability and calibration evaluation utilities.
"""

import numpy as np
import pytest

from foodspec.trust.reliability import (
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    reliability_curve_data,
    top_class_confidence,
)


class TestTopClassConfidence:
    """Test top_class_confidence helper function."""

    def test_basic_extraction(self):
        """Extract max probability from each sample."""
        proba = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
        result = top_class_confidence(proba)
        expected = np.array([0.7, 0.6, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiclass(self):
        """Handle multiclass (> 2 classes)."""
        proba = np.array([[0.1, 0.5, 0.4], [0.3, 0.2, 0.5]])
        result = top_class_confidence(proba)
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_perfect_confidence(self):
        """Handle near-perfect predictions."""
        proba = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = top_class_confidence(proba)
        expected = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_uniform_distribution(self):
        """Handle uniform prediction distribution."""
        # Valid case with uniform distribution
        proba_valid = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
        result = top_class_confidence(proba_valid)
        expected = np.array([1 / 3, 1 / 3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_input_shape(self):
        """Reject 1D or 3D input."""
        with pytest.raises(ValueError, match="2D"):
            top_class_confidence(np.array([0.7, 0.3]))
        with pytest.raises(ValueError, match="2D"):
            top_class_confidence(np.array([[[0.7, 0.3]]]))


class TestBrierScore:
    """Test Brier score computation."""

    def test_perfect_predictions(self):
        """Brier = 0 for perfect predictions."""
        y_true = [0, 1, 0, 1]
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        bs = brier_score(y_true, proba)
        assert bs < 1e-10  # Essentially 0

    def test_worst_predictions(self):
        """Brier is high for wrong predictions."""
        y_true = [0, 1, 0, 1]
        proba = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        bs = brier_score(y_true, proba)
        assert bs > 0.9  # Close to 1 (worst for binary)

    def test_uniform_probabilities(self):
        """Brier for uniform (uncertain) predictions."""
        y_true = [0, 1, 0, 1]
        proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        bs = brier_score(y_true, proba)
        # For binary uniform: (0.5 - 1)^2 + (0.5 - 0)^2 = 0.25 + 0.25 = 0.5
        # But sum of all squared errors: mean((0.5, 0.5) - (1, 0))^2 = mean(0.5^2 + 0.5^2) = 0.5
        # Actually: mean((proba - one_hot)^2) = mean([[-0.5, 0.5]^2, [0.5, -0.5]^2, ...])
        #         = mean([0.25 + 0.25, 0.25 + 0.25, ...]) = 0.5
        # No wait: ([0.5, 0.5] - [1, 0])^2 = [-0.5, 0.5]^2 = [0.25, 0.25], mean = 0.25
        # So Brier = 0.25 for all uniform predictions
        assert np.isclose(bs, 0.25)

    def test_multiclass_example(self):
        """Compute Brier for multiclass."""
        y_true = [0, 1, 2]
        proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        bs = brier_score(y_true, proba)
        assert bs < 1e-10

    def test_known_example(self):
        """Verify with known computed example."""
        y_true = [0, 1, 0, 1]
        proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.25, 0.75]])
        bs = brier_score(y_true, proba)
        # Formula: mean((proba - one_hot)^2)
        # Sample 0: proba=[0.9, 0.1], y=0, one_hot=[1, 0]
        #   error = ([0.9, 0.1] - [1, 0])^2 = [-0.1, 0.1]^2 = [0.01, 0.01], mean = 0.01
        # Sample 1: proba=[0.2, 0.8], y=1, one_hot=[0, 1]
        #   error = ([0.2, 0.8] - [0, 1])^2 = [0.2, -0.2]^2 = [0.04, 0.04], mean = 0.04
        # Sample 2: proba=[0.85, 0.15], y=0, one_hot=[1, 0]
        #   error = ([0.85, 0.15] - [1, 0])^2 = [-0.15, 0.15]^2 = [0.0225, 0.0225], mean = 0.0225
        # Sample 3: proba=[0.25, 0.75], y=1, one_hot=[0, 1]
        #   error = ([0.25, 0.75] - [0, 1])^2 = [0.25, -0.25]^2 = [0.0625, 0.0625], mean = 0.0625
        # Overall Brier = mean(0.01, 0.04, 0.0225, 0.0625) = 0.13375 / 4 = 0.033375
        expected = (0.01 + 0.04 + 0.0225 + 0.0625) / 4
        assert np.isclose(bs, expected)

    def test_invalid_labels(self):
        """Reject invalid label indices."""
        y_true = [0, 1, 2]  # Class 2 doesn't exist
        proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="out of range"):
            brier_score(y_true, proba)

    def test_shape_mismatch(self):
        """Reject shape mismatches."""
        y_true = [0, 1]
        proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="same number"):
            brier_score(y_true, proba)


class TestReliabilityCurveData:
    """Test reliability curve data computation."""

    def test_uniform_strategy(self):
        """Test uniform binning strategy."""
        y_true = [0, 0, 0, 1, 1, 1]
        proba = np.array([
            [0.8, 0.2],
            [0.9, 0.1],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ])
        centers, accs, confs, counts = reliability_curve_data(y_true, proba, n_bins=2)
        assert len(centers) == 2
        assert len(accs) == 2
        assert len(confs) == 2
        assert len(counts) == 2
        assert counts.sum() == 6

    def test_quantile_strategy(self):
        """Test quantile binning strategy."""
        y_true = [0, 0, 0, 1, 1, 1]
        proba = np.array([
            [0.8, 0.2],
            [0.9, 0.1],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ])
        centers, accs, confs, counts = reliability_curve_data(
            y_true, proba, n_bins=2, strategy="quantile"
        )
        assert len(centers) == 2
        assert counts.sum() == 6

    def test_bin_centers_match_confidences(self):
        """Bin centers should match confidences in populated bins."""
        y_true = [0, 1]
        proba = np.array([[0.8, 0.2], [0.3, 0.7]])
        centers, accs, confs, counts = reliability_curve_data(y_true, proba, n_bins=2)
        # Non-empty bins should have centers == confidences
        for i in range(len(centers)):
            if counts[i] > 0:
                assert np.isclose(centers[i], confs[i])

    def test_accuracy_bounds(self):
        """Accuracy should be in [0, 1]."""
        y_true = np.random.randint(0, 3, 100)
        proba = np.random.dirichlet(np.ones(3), size=100)
        centers, accs, confs, counts = reliability_curve_data(y_true, proba)
        assert np.all(accs >= 0)
        assert np.all(accs <= 1)

    def test_empty_bins_have_zero_accuracy(self):
        """Empty bins should report 0 accuracy."""
        y_true = [0, 0, 0, 0]
        proba = np.array([[0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.96, 0.04]])
        centers, accs, confs, counts = reliability_curve_data(y_true, proba, n_bins=10)
        # Most bins should be empty, having 0 accuracy
        empty_mask = counts == 0
        assert np.all(accs[empty_mask] == 0)


class TestExpectedCalibrationError:
    """Test Expected Calibration Error."""

    def test_perfect_calibration(self):
        """ECE should be low for well-calibrated predictions."""
        # All high-confidence predictions are correct - this should have low ECE
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        proba = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        ece = expected_calibration_error(y_true, proba)
        # All samples have 0.9 confidence, all are correct (accuracy = 1.0)
        # With n_bins=15 and 10 samples, bins might have 0 or 1 sample each
        # But logically: confidence=0.9, accuracy=1.0, so calibration gap=0.1
        # Single occupied bin: ECE = |accuracy - confidence| * (count/total) = 0.1 * 1.0 = 0.1
        assert ece <= 0.15  # Allow some tolerance

    def test_worst_calibration(self):
        """ECE should be high for poorly calibrated predictions."""
        # High-confidence predictions that are all wrong
        y_true = np.array([0, 0, 0, 0])
        proba = np.array([
            [0.1, 0.9],  # Predicts 1 with 0.9 confidence, true is 0
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
        ])
        ece = expected_calibration_error(y_true, proba)
        assert ece > 0.5  # High miscalibration

    def test_uniform_strategy_vs_quantile(self):
        """Both strategies should produce valid ECE."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        proba = np.random.dirichlet(np.ones(2), size=100)

        ece_uniform = expected_calibration_error(y_true, proba, strategy="uniform")
        ece_quantile = expected_calibration_error(y_true, proba, strategy="quantile")

        assert 0 <= ece_uniform <= 1
        assert 0 <= ece_quantile <= 1

    def test_deterministic(self):
        """ECE computation should be deterministic (no randomness)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 50)
        proba = np.random.dirichlet(np.ones(3), size=50)

        ece1 = expected_calibration_error(y_true, proba)
        ece2 = expected_calibration_error(y_true, proba)

        assert ece1 == ece2

    def test_multiclass_ece(self):
        """ECE should work for multiclass problems."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        proba = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        ece = expected_calibration_error(y_true, proba)
        # Top-class confidence = [0.7, 0.8, 0.8, 0.7, 0.8, 0.8]
        # All predictions are correct (accuracy = 1.0)
        # Calibration gap for 0.7-confidence: |1.0 - 0.7| = 0.3
        # Calibration gap for 0.8-confidence: |1.0 - 0.8| = 0.2
        # With binning this could vary, so just check reasonable range
        assert 0 <= ece <= 0.5


class TestComputeCalibrationMetrics:
    """Test combined calibration metrics computation."""

    def test_returns_valid_metrics(self):
        """Should return CalibrationMetrics with all fields populated."""
        y_true = [0, 1, 0, 1]
        proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8]])
        metrics = compute_calibration_metrics(y_true, proba)

        assert 0 <= metrics.ece <= 1
        assert 0 <= metrics.brier <= 2
        assert len(metrics.bin_centers) > 0
        assert len(metrics.accuracies) > 0
        assert len(metrics.counts) > 0
        assert len(metrics.bin_centers) == len(metrics.accuracies)

    def test_perfect_predictions_perfect_metrics(self):
        """Perfect predictions should have low ECE and Brier."""
        y_true = [0, 1, 0, 1]
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        metrics = compute_calibration_metrics(y_true, proba)

        assert metrics.ece < 0.01
        assert metrics.brier < 0.01

    def test_deterministic_computation(self):
        """Metrics should be deterministic."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        proba = np.random.dirichlet(np.ones(2), size=50)

        metrics1 = compute_calibration_metrics(y_true, proba)
        metrics2 = compute_calibration_metrics(y_true, proba)

        assert metrics1.ece == metrics2.ece
        assert metrics1.brier == metrics2.brier


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_class_prediction(self):
        """Handle single-class binary predictions (all class 0 or all class 1)."""
        y_true = [0, 0, 0, 0]
        proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.95, 0.05]])
        ece = expected_calibration_error(y_true, proba)
        assert 0 <= ece <= 1

    def test_few_samples(self):
        """Handle small number of samples."""
        y_true = [0, 1]
        proba = np.array([[0.7, 0.3], [0.2, 0.8]])
        ece = expected_calibration_error(y_true, proba, n_bins=2)
        assert 0 <= ece <= 1

    def test_more_bins_than_samples(self):
        """Handle n_bins > n_samples."""
        y_true = [0, 1]
        proba = np.array([[0.7, 0.3], [0.2, 0.8]])
        ece = expected_calibration_error(y_true, proba, n_bins=100)
        assert 0 <= ece <= 1

    def test_negative_labels_rejected(self):
        """Reject negative class labels."""
        y_true = [-1, 0, 1]
        proba = np.array([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4], [0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="out of range"):
            expected_calibration_error(y_true, proba)

    def test_invalid_strategy(self):
        """Reject invalid binning strategy."""
        y_true = [0, 1]
        proba = np.array([[0.7, 0.3], [0.2, 0.8]])
        with pytest.raises(ValueError, match="Unknown strategy"):
            reliability_curve_data(y_true, proba, strategy="invalid")
