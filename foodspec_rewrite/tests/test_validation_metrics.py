"""Tests for validation metrics.

Tests classification and calibration metrics including:
- accuracy, precision, recall, F1
- AUROC (one-vs-rest, macro-averaged)
- Expected Calibration Error (ECE)
- Edge cases: missing classes in folds (LOBO), single class, missing probabilities
"""

import numpy as np
import pytest

from foodspec.validation.metrics import (
    accuracy,
    auroc_macro,
    compute_classification_metrics,
    expected_calibration_error,
    macro_f1,
    precision_macro,
    recall_macro,
)


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_perfect_accuracy(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        result = accuracy(y_true, y_pred)

        assert result == {"accuracy": 1.0}

    def test_zero_accuracy(self):
        """Test accuracy with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 2, 0, 2, 0, 1])

        result = accuracy(y_true, y_pred)

        assert result == {"accuracy": 0.0}

    def test_partial_accuracy(self):
        """Test accuracy with some correct predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])

        result = accuracy(y_true, y_pred)

        assert result == {"accuracy": 0.75}

    def test_binary_classification(self):
        """Test accuracy for binary classification."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])

        result = accuracy(y_true, y_pred)

        assert result == {"accuracy": 2 / 3}

    def test_length_mismatch(self):
        """Test error when y_true and y_pred have different lengths."""
        y_true = np.array([0, 0, 1])
        y_pred = np.array([0, 0])

        with pytest.raises(ValueError, match="same length"):
            accuracy(y_true, y_pred)

    def test_wrong_dimensions(self):
        """Test error for non-1D inputs."""
        y_true = np.array([[0, 0], [1, 1]])
        y_pred = np.array([0, 0])

        with pytest.raises(ValueError, match="1D array"):
            accuracy(y_true, y_pred)


class TestMacroF1:
    """Tests for macro-averaged F1 score."""

    def test_perfect_f1(self):
        """Test F1 with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        result = macro_f1(y_true, y_pred)

        assert result == {"macro_f1": 1.0}

    def test_partial_f1(self):
        """Test F1 with some correct predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])

        result = macro_f1(y_true, y_pred)

        # Class 0: TP=2, FP=1, FN=0 -> precision=2/3, recall=1.0, F1=0.8
        # Class 1: TP=1, FP=0, FN=1 -> precision=1.0, recall=0.5, F1=0.667
        # Macro F1 = (0.8 + 0.667) / 2 = 0.733
        expected_f1 = pytest.approx(0.733, abs=0.01)
        assert result["macro_f1"] == expected_f1

    def test_binary_f1(self):
        """Test F1 for binary classification."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])

        result = macro_f1(y_true, y_pred)

        # Class 0: TP=2, FP=1, FN=1 -> F1=0.667
        # Class 1: TP=2, FP=1, FN=1 -> F1=0.667
        # Macro F1 = 0.667
        assert result["macro_f1"] == pytest.approx(0.667, abs=0.01)

    def test_missing_classes_in_predictions(self):
        """Test F1 when a class is never predicted (common in LOBO)."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 1])  # Never predicts class 2

        result = macro_f1(y_true, y_pred)

        # Should handle this gracefully with zero_division=0
        assert "macro_f1" in result
        assert 0.0 <= result["macro_f1"] <= 1.0


class TestPrecisionMacro:
    """Tests for macro-averaged precision."""

    def test_perfect_precision(self):
        """Test precision with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        result = precision_macro(y_true, y_pred)

        assert result == {"precision_macro": 1.0}

    def test_partial_precision(self):
        """Test precision with some errors."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])

        result = precision_macro(y_true, y_pred)

        # Class 0: TP=2, FP=1 -> precision=2/3
        # Class 1: TP=1, FP=0 -> precision=1.0
        # Macro precision = (2/3 + 1.0) / 2 = 5/6
        assert result["precision_macro"] == pytest.approx(5 / 6, abs=0.001)

    def test_zero_division_handling(self):
        """Test precision when a class is never predicted."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])  # Never predicts class 1

        result = precision_macro(y_true, y_pred)

        # Should handle zero division gracefully
        assert "precision_macro" in result
        assert 0.0 <= result["precision_macro"] <= 1.0


class TestRecallMacro:
    """Tests for macro-averaged recall."""

    def test_perfect_recall(self):
        """Test recall with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        result = recall_macro(y_true, y_pred)

        assert result == {"recall_macro": 1.0}

    def test_partial_recall(self):
        """Test recall with some errors."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])

        result = recall_macro(y_true, y_pred)

        # Class 0: TP=2, FN=0 -> recall=1.0
        # Class 1: TP=1, FN=1 -> recall=0.5
        # Macro recall = (1.0 + 0.5) / 2 = 0.75
        assert result["recall_macro"] == 0.75


class TestAUROCMacro:
    """Tests for macro-averaged AUROC (one-vs-rest)."""

    def test_perfect_auroc(self):
        """Test AUROC with perfect probability predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

        result = auroc_macro(y_true, None, y_proba)

        assert result == {"auroc_macro": 1.0}

    def test_random_auroc(self):
        """Test AUROC with random-like predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = auroc_macro(y_true, None, y_proba)

        # Random predictions should give AUROC ~ 0.5
        assert result["auroc_macro"] == 0.5

    def test_multiclass_auroc(self):
        """Test AUROC for multiclass classification."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.2, 0.7],
            ]
        )

        result = auroc_macro(y_true, None, y_proba)

        assert "auroc_macro" in result
        assert result["auroc_macro"] > 0.9  # Should be high for well-separated classes

    def test_missing_class_in_fold(self):
        """Test AUROC when validation fold is missing a class (LOBO scenario)."""
        # Training set has 3 classes, but this fold only has 2
        y_true = np.array([0, 0, 1, 1])
        # Probabilities include class 2 (never seen in this fold)
        y_proba = np.array(
            [[0.7, 0.2, 0.1], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]]
        )

        result = auroc_macro(y_true, None, y_proba)

        # Should handle gracefully and compute AUROC only for present classes
        assert "auroc_macro" in result
        assert not np.isnan(result["auroc_macro"])

    def test_single_class_returns_nan(self):
        """Test AUROC returns NaN when only one class present."""
        y_true = np.array([0, 0, 0, 0])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])

        result = auroc_macro(y_true, None, y_proba)

        assert "auroc_macro" in result
        assert np.isnan(result["auroc_macro"])

    def test_missing_proba_raises_error(self):
        """Test error when y_proba is not provided."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="y_proba is required"):
            auroc_macro(y_true, y_pred, None)

    def test_wrong_proba_shape(self):
        """Test error when y_proba has wrong shape."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1])  # 1D instead of 2D

        with pytest.raises(ValueError, match="2D array"):
            auroc_macro(y_true, None, y_proba)

    def test_proba_length_mismatch(self):
        """Test error when y_proba length doesn't match y_true."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2]])  # Only 2 samples

        with pytest.raises(ValueError, match="same length"):
            auroc_macro(y_true, None, y_proba)


class TestExpectedCalibrationError:
    """Tests for Expected Calibration Error (ECE)."""

    def test_perfect_calibration(self):
        """Test ECE with perfectly calibrated predictions."""
        # Perfect calibration: all predictions are correct
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        result = expected_calibration_error(y_true, None, y_proba)

        # Perfect predictions should have ECE close to 0
        assert result["ece"] == pytest.approx(0.0, abs=0.01)

    def test_poor_calibration(self):
        """Test ECE with poorly calibrated predictions."""
        # Predictions are correct but with wrong confidence
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.4, 0.6]])

        result = expected_calibration_error(y_true, None, y_proba)

        # Poor calibration should have ECE > 0
        assert result["ece"] > 0.0

    def test_different_bin_counts(self):
        """Test ECE with different number of bins."""
        y_true = np.array([0, 0, 1, 1] * 5)  # Repeat for more samples
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]] * 5)

        result_5_bins = expected_calibration_error(y_true, None, y_proba, n_bins=5)
        result_10_bins = expected_calibration_error(y_true, None, y_proba, n_bins=10)

        # Both should compute ECE, but values may differ
        assert "ece" in result_5_bins
        assert "ece" in result_10_bins
        assert result_5_bins["ece"] >= 0.0
        assert result_10_bins["ece"] >= 0.0

    def test_missing_proba_raises_error(self):
        """Test error when y_proba is not provided."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="y_proba is required"):
            expected_calibration_error(y_true, y_pred, None)

    def test_invalid_bins(self):
        """Test error for invalid number of bins."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])

        with pytest.raises(ValueError, match="n_bins must be positive"):
            expected_calibration_error(y_true, None, y_proba, n_bins=0)

        with pytest.raises(ValueError, match="n_bins must be positive"):
            expected_calibration_error(y_true, None, y_proba, n_bins=-1)

    def test_multiclass_ece(self):
        """Test ECE for multiclass classification."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.2, 0.7],
            ]
        )

        result = expected_calibration_error(y_true, None, y_proba)

        assert "ece" in result
        assert result["ece"] >= 0.0
        assert result["ece"] <= 1.0


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics convenience function."""

    def test_all_metrics_without_proba(self):
        """Test computing metrics without probabilities."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])

        result = compute_classification_metrics(y_true, y_pred, y_proba=None)

        # Should have basic metrics
        assert "accuracy" in result
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "macro_f1" in result

        # Should not have probability-based metrics
        assert "auroc_macro" not in result
        assert "ece" not in result

    def test_all_metrics_with_proba(self):
        """Test computing all metrics with probabilities."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        # Should have all metrics
        assert "accuracy" in result
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "macro_f1" in result
        assert "auroc_macro" in result
        assert "ece" in result

        # Check values are in valid ranges
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["precision_macro"] <= 1.0
        assert 0.0 <= result["recall_macro"] <= 1.0
        assert 0.0 <= result["macro_f1"] <= 1.0
        assert 0.0 <= result["auroc_macro"] <= 1.0
        assert 0.0 <= result["ece"] <= 1.0

    def test_skip_ece(self):
        """Test computing metrics without ECE."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])

        result = compute_classification_metrics(
            y_true, y_pred, y_proba, include_ece=False
        )

        # Should have most metrics
        assert "accuracy" in result
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "macro_f1" in result
        assert "auroc_macro" in result

        # Should not have ECE
        assert "ece" not in result

    def test_custom_ece_bins(self):
        """Test ECE with custom number of bins."""
        y_true = np.array([0, 0, 1, 1] * 10)
        y_pred = np.array([0, 0, 1, 0] * 10)
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]] * 10)

        result = compute_classification_metrics(y_true, y_pred, y_proba, ece_bins=5)

        assert "ece" in result

    def test_multiclass_all_metrics(self):
        """Test all metrics for multiclass classification."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 1])
        y_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.6, 0.3, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.6, 0.3],
            ]
        )

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        # Should have all metrics
        assert len(result) == 6
        assert "accuracy" in result
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "macro_f1" in result
        assert "auroc_macro" in result
        assert "ece" in result

    def test_missing_class_in_fold(self):
        """Test handling of missing classes (LOBO scenario)."""
        # Fold only has classes 0 and 1, but model was trained on 0, 1, 2
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        # Probabilities include class 2
        y_proba = np.array(
            [[0.7, 0.2, 0.1], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1]]
        )

        result = compute_classification_metrics(y_true, y_pred, y_proba)

        # Should compute all metrics without errors
        assert "accuracy" in result
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "macro_f1" in result
        assert "auroc_macro" in result
        assert "ece" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        # sklearn might return NaN or raise an error depending on the metric
        result = accuracy(y_true, y_pred)
        # Empty arrays should return NaN for accuracy
        assert np.isnan(result["accuracy"]) or result["accuracy"] == 0.0

    def test_single_sample(self):
        """Test with single sample."""
        y_true = np.array([0])
        y_pred = np.array([0])

        result = accuracy(y_true, y_pred)

        assert result == {"accuracy": 1.0}

    def test_large_number_of_classes(self):
        """Test with many classes."""
        n_samples = 100
        n_classes = 20

        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)

        # Should compute without errors
        result = accuracy(y_true, y_pred)

        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0
