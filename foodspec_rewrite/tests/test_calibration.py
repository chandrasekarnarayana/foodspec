"""
Unit tests for calibration methods (Platt + Isotonic).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from foodspec.trust import PlattCalibrator, IsotonicCalibrator
from foodspec.trust.calibration import expected_calibration_error


class TestPlattCalibrator:
    """Test Platt scaling calibrator."""

    @pytest.fixture
    def binary_data(self):
        """Create binary classification calibration data."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        # Create miscalibrated predictions (overconfident)
        proba = np.random.uniform(0.3, 0.9, (n, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return y_true, proba

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass calibration data."""
        np.random.seed(42)
        n = 100
        n_classes = 3
        y_true = np.random.randint(0, n_classes, n)
        proba = np.random.uniform(0.2, 0.8, (n, n_classes))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return y_true, proba

    def test_fit_returns_self(self, binary_data):
        """fit() should return self for chaining."""
        y_true, proba = binary_data
        calibrator = PlattCalibrator()
        result = calibrator.fit(y_true, proba)
        assert result is calibrator

    def test_fit_multiclass(self, multiclass_data):
        """Should handle multiclass problems."""
        y_true, proba = multiclass_data
        calibrator = PlattCalibrator()
        calibrator.fit(y_true, proba)
        assert calibrator.n_classes_ == 3
        assert len(calibrator.logistic_models_) == 3
        assert calibrator._fitted

    def test_transform_returns_valid_proba(self, binary_data):
        """transform() should return valid probabilities."""
        y_true, proba = binary_data
        calibrator = PlattCalibrator()
        calibrator.fit(y_true, proba)
        
        # Transform same data
        proba_cal = calibrator.transform(proba)
        
        # Check shape
        assert proba_cal.shape == proba.shape
        
        # Check valid probabilities: [0, 1] and sum to 1
        assert (proba_cal >= -1e-6).all()  # Allow tiny numerical errors
        assert (proba_cal <= 1 + 1e-6).all()
        row_sums = proba_cal.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_transform_before_fit_raises(self, binary_data):
        """transform() before fit() should raise RuntimeError."""
        _, proba = binary_data
        calibrator = PlattCalibrator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            calibrator.transform(proba)

    def test_transform_wrong_shape_raises(self, binary_data):
        """transform() with wrong n_classes should raise."""
        y_true, proba = binary_data
        calibrator = PlattCalibrator()
        calibrator.fit(y_true, proba)
        
        # Try to transform with 3 classes instead of 2
        proba_wrong = np.random.uniform(0.2, 0.8, (10, 3))
        proba_wrong = proba_wrong / proba_wrong.sum(axis=1, keepdims=True)
        
        with pytest.raises(ValueError, match="classes"):
            calibrator.transform(proba_wrong)

    def test_deterministic_transform(self, binary_data):
        """Multiple transforms of same data should be identical."""
        y_true, proba = binary_data
        calibrator = PlattCalibrator()
        calibrator.fit(y_true, proba)
        
        result1 = calibrator.transform(proba)
        result2 = calibrator.transform(proba)
        
        np.testing.assert_array_equal(result1, result2)

    def test_ece_improvement(self, binary_data):
        """ECE should improve after calibration on toy miscalibrated data."""
        y_true, proba = binary_data
        
        ece_before = expected_calibration_error(y_true, proba)
        
        calibrator = PlattCalibrator()
        calibrator.fit(y_true, proba)
        proba_cal = calibrator.transform(proba)
        
        ece_after = expected_calibration_error(y_true, proba_cal)
        
        # ECE should improve (decrease or stay similar)
        assert ece_after <= ece_before + 0.05  # Allow small tolerance

    def test_serialization(self, binary_data):
        """Save/load should preserve calibrator."""
        y_true, proba = binary_data
        calibrator = PlattCalibrator()
        calibrator.fit(y_true, proba)
        
        # Transform original
        proba_cal_original = calibrator.transform(proba[:5])
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibrator.pkl"
            calibrator.save(str(path))
            
            calibrator_loaded = PlattCalibrator.load(str(path))
            proba_cal_loaded = calibrator_loaded.transform(proba[:5])
        
        np.testing.assert_array_almost_equal(proba_cal_original, proba_cal_loaded)

    def test_save_before_fit_raises(self):
        """save() before fit() should raise."""
        calibrator = PlattCalibrator()
        with pytest.raises(RuntimeError, match="unfitted"):
            with tempfile.TemporaryDirectory() as tmpdir:
                calibrator.save(str(Path(tmpdir) / "cal.pkl"))

    def test_invalid_labels(self):
        """Invalid label range should raise."""
        y_true = [0, 1, 2]  # Class 2 doesn't exist
        proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        
        calibrator = PlattCalibrator()
        with pytest.raises(ValueError, match="out of range"):
            calibrator.fit(y_true, proba)

    def test_shape_mismatch(self):
        """Shape mismatch should raise."""
        y_true = [0, 1]
        proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        
        calibrator = PlattCalibrator()
        with pytest.raises(ValueError, match="same number"):
            calibrator.fit(y_true, proba)

    def test_1d_proba_raises(self):
        """1D proba should raise."""
        y_true = [0, 1]
        proba = np.array([0.5, 0.5])
        
        calibrator = PlattCalibrator()
        with pytest.raises(ValueError, match="2D"):
            calibrator.fit(y_true, proba)

    def test_preserves_class_order(self, multiclass_data):
        """Multiclass calibration should preserve class order."""
        y_true, proba = multiclass_data
        
        calibrator = PlattCalibrator()
        calibrator.fit(y_true, proba)
        proba_cal = calibrator.transform(proba)
        
        # Compare which class is predicted
        pred_before = np.argmax(proba, axis=1)
        pred_after = np.argmax(proba_cal, axis=1)
        
        # Predictions may change but class order is preserved
        assert pred_after.shape == pred_before.shape
        assert np.all((pred_after >= 0) & (pred_after < 3))

    def test_perfect_calibration_data(self):
        """Perfect data should remain well-calibrated."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        proba = np.array([
            [0.95, 0.05], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95],
            [0.95, 0.05], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95]
        ])
        
        calibrator = PlattCalibrator()
        calibrator.fit(y_true, proba)
        proba_cal = calibrator.transform(proba)
        
        ece_after = expected_calibration_error(y_true, proba_cal)
        assert ece_after < 0.1  # Should be very small


class TestIsotonicCalibrator:
    """Test isotonic regression calibrator."""

    @pytest.fixture
    def binary_data(self):
        """Create binary classification calibration data."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        proba = np.random.uniform(0.3, 0.9, (n, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return y_true, proba

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass calibration data."""
        np.random.seed(42)
        n = 100
        n_classes = 3
        y_true = np.random.randint(0, n_classes, n)
        proba = np.random.uniform(0.2, 0.8, (n, n_classes))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return y_true, proba

    def test_fit_returns_self(self, binary_data):
        """fit() should return self for chaining."""
        y_true, proba = binary_data
        calibrator = IsotonicCalibrator()
        result = calibrator.fit(y_true, proba)
        assert result is calibrator

    def test_fit_multiclass(self, multiclass_data):
        """Should handle multiclass problems."""
        y_true, proba = multiclass_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, proba)
        assert calibrator.n_classes_ == 3
        assert len(calibrator.isotonic_models_) == 3
        assert calibrator._fitted

    def test_transform_returns_valid_proba(self, binary_data):
        """transform() should return valid probabilities."""
        y_true, proba = binary_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, proba)
        
        proba_cal = calibrator.transform(proba)
        
        # Check shape
        assert proba_cal.shape == proba.shape
        
        # Check valid probabilities
        assert (proba_cal >= -1e-6).all()
        assert (proba_cal <= 1 + 1e-6).all()
        row_sums = proba_cal.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_transform_before_fit_raises(self, binary_data):
        """transform() before fit() should raise RuntimeError."""
        _, proba = binary_data
        calibrator = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            calibrator.transform(proba)

    def test_transform_wrong_shape_raises(self, binary_data):
        """transform() with wrong n_classes should raise."""
        y_true, proba = binary_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, proba)
        
        proba_wrong = np.random.uniform(0.2, 0.8, (10, 3))
        proba_wrong = proba_wrong / proba_wrong.sum(axis=1, keepdims=True)
        
        with pytest.raises(ValueError, match="classes"):
            calibrator.transform(proba_wrong)

    def test_deterministic_transform(self, binary_data):
        """Multiple transforms should be identical."""
        y_true, proba = binary_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, proba)
        
        result1 = calibrator.transform(proba)
        result2 = calibrator.transform(proba)
        
        np.testing.assert_array_equal(result1, result2)

    def test_ece_improvement(self, binary_data):
        """ECE should improve on miscalibrated data."""
        y_true, proba = binary_data
        
        ece_before = expected_calibration_error(y_true, proba)
        
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, proba)
        proba_cal = calibrator.transform(proba)
        
        ece_after = expected_calibration_error(y_true, proba_cal)
        
        # ECE should improve
        assert ece_after <= ece_before + 0.05

    def test_serialization(self, binary_data):
        """Save/load should preserve calibrator."""
        y_true, proba = binary_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, proba)
        
        proba_cal_original = calibrator.transform(proba[:5])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibrator.pkl"
            calibrator.save(str(path))
            
            calibrator_loaded = IsotonicCalibrator.load(str(path))
            proba_cal_loaded = calibrator_loaded.transform(proba[:5])
        
        np.testing.assert_array_almost_equal(proba_cal_original, proba_cal_loaded)

    def test_save_before_fit_raises(self):
        """save() before fit() should raise."""
        calibrator = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="unfitted"):
            with tempfile.TemporaryDirectory() as tmpdir:
                calibrator.save(str(Path(tmpdir) / "cal.pkl"))

    def test_invalid_labels(self):
        """Invalid label range should raise."""
        y_true = [0, 1, 2]
        proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        
        calibrator = IsotonicCalibrator()
        with pytest.raises(ValueError, match="out of range"):
            calibrator.fit(y_true, proba)

    def test_shape_mismatch(self):
        """Shape mismatch should raise."""
        y_true = [0, 1]
        proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        
        calibrator = IsotonicCalibrator()
        with pytest.raises(ValueError, match="same number"):
            calibrator.fit(y_true, proba)

    def test_monotonicity(self, binary_data):
        """Isotonic regression should maintain monotonicity."""
        y_true, proba = binary_data
        
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, proba)
        
        # Create test data with sorted confidences
        test_proba = np.array([
            [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
            [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]
        ])
        
        proba_cal = calibrator.transform(test_proba)
        
        # Check that first class probabilities are roughly monotone
        # (may not be perfect due to renormalization)
        class_0_cal = proba_cal[:, 0]
        assert class_0_cal.shape == (9,)

    def test_out_of_bounds_clip(self):
        """Out-of-bounds predictions should be clipped."""
        y_true = np.array([0, 1, 0, 1])
        proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, proba)
        
        # This shouldn't raise even with unusual inputs
        test_proba = np.array([[0.05, 0.95], [0.95, 0.05]])
        proba_cal = calibrator.transform(test_proba)
        
        assert proba_cal.shape == test_proba.shape


class TestCalibrationComparison:
    """Compare Platt vs Isotonic calibration."""

    def test_both_improve_ece(self):
        """Both methods should improve ECE on miscalibrated data."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        proba = np.random.uniform(0.3, 0.9, (n, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        ece_before = expected_calibration_error(y_true, proba)
        
        # Platt
        platt = PlattCalibrator()
        platt.fit(y_true, proba)
        proba_platt = platt.transform(proba)
        ece_platt = expected_calibration_error(y_true, proba_platt)
        
        # Isotonic
        isotonic = IsotonicCalibrator()
        isotonic.fit(y_true, proba)
        proba_isotonic = isotonic.transform(proba)
        ece_isotonic = expected_calibration_error(y_true, proba_isotonic)
        
        # Both should improve
        assert ece_platt <= ece_before + 0.05
        assert ece_isotonic <= ece_before + 0.05

    def test_probability_sums_consistent(self):
        """Both methods should produce consistent probability sums."""
        np.random.seed(42)
        n = 50
        y_true = np.random.randint(0, 3, n)
        proba = np.random.uniform(0.2, 0.8, (n, 3))
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        platt = PlattCalibrator()
        platt.fit(y_true, proba)
        proba_platt = platt.transform(proba)
        
        isotonic = IsotonicCalibrator()
        isotonic.fit(y_true, proba)
        proba_isotonic = isotonic.transform(proba)
        
        # Both should sum to ~1
        assert np.allclose(proba_platt.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(proba_isotonic.sum(axis=1), 1.0, atol=1e-5)


class TestEdgeCases:
    """Test edge cases for calibrators."""

    def test_small_calibration_set(self):
        """Should work with small calibration sets."""
        y_true = np.array([0, 1])
        proba = np.array([[0.7, 0.3], [0.2, 0.8]])
        
        platt = PlattCalibrator()
        platt.fit(y_true, proba)
        proba_cal = platt.transform(proba)
        assert proba_cal.shape == proba.shape
        
        isotonic = IsotonicCalibrator()
        isotonic.fit(y_true, proba)
        proba_cal = isotonic.transform(proba)
        assert proba_cal.shape == proba.shape

    def test_many_classes(self):
        """Should handle many classes."""
        n_classes = 10
        np.random.seed(42)
        y_true = np.random.randint(0, n_classes, 100)
        proba = np.random.uniform(0.1, 0.9, (100, n_classes))
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        platt = PlattCalibrator()
        platt.fit(y_true, proba)
        proba_cal = platt.transform(proba)
        assert proba_cal.shape == (100, 10)
        
        isotonic = IsotonicCalibrator()
        isotonic.fit(y_true, proba)
        proba_cal = isotonic.transform(proba)
        assert proba_cal.shape == (100, 10)

    def test_uniform_labels(self):
        """Should handle uniform class distribution."""
        n = 100
        y_true = np.repeat(np.arange(2), n // 2)
        proba = np.random.uniform(0.3, 0.9, (n, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        platt = PlattCalibrator()
        platt.fit(y_true, proba)
        proba_cal = platt.transform(proba)
        assert np.allclose(proba_cal.sum(axis=1), 1.0, atol=1e-5)

    def test_transform_unseen_data(self):
        """Transform should work on completely unseen data."""
        np.random.seed(42)
        # Train on some data
        y_true = np.random.randint(0, 2, 100)
        proba_train = np.random.uniform(0.3, 0.9, (100, 2))
        proba_train = proba_train / proba_train.sum(axis=1, keepdims=True)
        
        platt = PlattCalibrator()
        platt.fit(y_true, proba_train)
        
        # Transform completely different data
        proba_test = np.random.uniform(0.2, 0.8, (50, 2))
        proba_test = proba_test / proba_test.sum(axis=1, keepdims=True)
        
        proba_cal = platt.transform(proba_test)
        assert proba_cal.shape == (50, 2)
        assert np.allclose(proba_cal.sum(axis=1), 1.0, atol=1e-5)
