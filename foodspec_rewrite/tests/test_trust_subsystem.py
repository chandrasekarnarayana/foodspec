"""
Comprehensive tests for trust and uncertainty quantification subsystem.

Tests cover:
- Conformal prediction with bin conditioning (Mondrian)
- Abstention rules and performance metrics
- Calibration methods (temperature scaling, isotonic)
- Group-aware coverage computation
- Determinism with seeds

Uses new API with probability arrays directly.
"""

import numpy as np
import pandas as pd
import pytest

from foodspec.trust.conformal import MondrianConformalClassifier, ConformalPredictionResult
from foodspec.trust.abstain import evaluate_abstention, AbstentionResult
from foodspec.trust.calibration import (
    expected_calibration_error,
    TemperatureScaler,
    IsotonicCalibrator,
)


class TestMondrianConformal:
    """Test Mondrian conformal prediction."""
    
    @pytest.fixture
    def binary_data(self):
        """Generate synthetic binary classification data with probabilities."""
        np.random.seed(42)
        n_cal = 100
        n_test = 50
        
        y_cal = np.random.randint(0, 2, n_cal)
        proba_cal = np.random.dirichlet([1, 1], size=n_cal)
        
        y_test = np.random.randint(0, 2, n_test)
        proba_test = np.random.dirichlet([1, 1], size=n_test)
        
        return y_cal, proba_cal, y_test, proba_test
    
    def test_fit_and_predict(self, binary_data):
        """Test basic fit and predict workflow."""
        y_cal, proba_cal, y_test, proba_test = binary_data
        
        # Initialize with alpha = 1 - target_coverage
        target_coverage = 0.9
        alpha = 1.0 - target_coverage
        cp = MondrianConformalClassifier(alpha=alpha)
        
        # Fit
        cp.fit(y_cal, proba_cal)
        assert cp._fitted
        assert cp._n_classes == 2
        
        # Predict
        result = cp.predict_sets(proba_test, y_true=y_test)
        
        assert isinstance(result, ConformalPredictionResult)
        assert len(result.prediction_sets) == len(y_test)
        assert len(result.set_sizes) == len(y_test)
        assert all(sz >= 1 for sz in result.set_sizes)
        assert result.coverage is not None
        assert 0 <= result.coverage <= 1
    
    def test_mondrian_with_bins(self, binary_data):
        """Test Mondrian binning for conditional coverage."""
        y_cal, proba_cal, y_test, proba_test = binary_data
        
        # Create bins based on uncertainty
        proba_max_cal = proba_cal.max(axis=1)
        meta_cal = np.array(['high_conf' if pm > 0.6 else 'low_conf' for pm in proba_max_cal])
        
        proba_max_test = proba_test.max(axis=1)
        meta_test = np.array(['high_conf' if pm > 0.6 else 'low_conf' for pm in proba_max_test])
        
        # Fit with conditioning
        cp = MondrianConformalClassifier(alpha=0.1, condition_key='conf', min_bin_size=10)
        cp.fit(y_cal, proba_cal, meta_cal=meta_cal)
        
        # Predict with bins
        result = cp.predict_sets(proba_test, meta_test=meta_test, y_true=y_test)
        
        # Should have coverage for each bin
        assert len(result.per_bin_coverage) > 0
        assert result.coverage is not None


class TestAbstention:
    """Test abstention utilities."""
    
    def test_abstention_by_confidence(self):
        """Test abstention based on confidence threshold."""
        proba = np.array([
            [0.9, 0.1],
            [0.55, 0.45],
            [0.7, 0.3],
            [0.4, 0.6],
        ])
        y_true = [0, 1, 0, 1]
        
        result = evaluate_abstention(proba, y_true, threshold=0.6)
        
        assert isinstance(result, AbstentionResult)
        assert len(result.abstain_mask) == 4
        assert result.abstain_mask[0] == False  # 0.9 >= 0.6
        assert result.abstain_mask[1] == True   # 0.55 < 0.6
        assert result.abstain_mask[3] == True   # 0.6 is not >= 0.6
    
    def test_abstention_accuracy_non_abstained(self):
        """Test accuracy on non-abstained samples."""
        proba = np.array([
            [0.9, 0.1],  # Correct, high conf
            [0.55, 0.45],  # Will abstain
            [0.7, 0.3],    # Correct, high conf
            [0.3, 0.7],    # Wrong pred, will abstain
        ])
        y_true = [0, 1, 0, 1]
        
        result = evaluate_abstention(proba, y_true, threshold=0.6)
        
        # Only first two are not abstained: both correct
        assert result.accuracy_non_abstained == 1.0
    
    def test_abstention_with_prediction_sets(self):
        """Test abstention with prediction set size constraint."""
        proba = np.array([
            [0.6, 0.4],
            [0.5, 0.5],
            [0.7, 0.3],
        ])
        y_true = [0, 1, 0]
        prediction_sets = [[0], [0, 1], [0]]  # Sizes: 1, 2, 1
        
        result = evaluate_abstention(
            proba, y_true,
            threshold=0.5,
            prediction_sets=prediction_sets,
            max_set_size=1,
        )
        
        # Second sample should abstain due to large set size
        assert result.abstain_mask[1] == True


class TestCalibration:
    """Test probability calibration."""
    
    def test_ece_perfect_calibration(self):
        """ECE should be ~0 for perfectly calibrated predictions."""
        # Perfect calibration: predicted prob = empirical accuracy
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([
            [0.9, 0.1],  # Correct, confidence = accuracy
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
        
        ece = expected_calibration_error(y_true, y_pred_proba, n_bins=5)
        assert ece < 0.05
    
    def test_temperature_scaler_fit_predict(self):
        """Test temperature scaling calibration."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100, 2)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
        
        scaler = TemperatureScaler()
        scaler.fit(y_true, y_pred_proba)
        
        assert 0.01 <= scaler.temperature <= 5.0
        
        scaled_proba = scaler.predict(y_pred_proba)
        assert scaled_proba.shape == y_pred_proba.shape
        assert np.allclose(scaled_proba.sum(axis=1), 1.0)
    
    def test_isotonic_calibrator_fit_transform(self):
        """Test isotonic regression calibration."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100, 2)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
        
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_true, y_pred_proba)
        
        calibrated = calibrator.transform(y_pred_proba)
        assert calibrated.shape == y_pred_proba.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)
        assert np.all((calibrated >= 0) & (calibrated <= 1))


class TestGroupAwareCoverage:
    """Test conditional coverage computation."""
    
    def test_conformal_per_bin_coverage(self):
        """Test per-bin coverage in conformal prediction."""
        np.random.seed(42)
        n_cal = 100
        n_test = 50
        
        # Calibration data with bins
        y_cal = np.random.randint(0, 2, n_cal)
        proba_cal = np.random.dirichlet([1, 1], size=n_cal)
        meta_cal = np.repeat([0, 1], n_cal // 2)
        
        # Test data
        y_test = np.random.randint(0, 2, n_test)
        proba_test = np.random.dirichlet([1, 1], size=n_test)
        meta_test = np.repeat([0, 1], n_test // 2)
        
        cp = MondrianConformalClassifier(alpha=0.1, condition_key='bin')
        cp.fit(y_cal, proba_cal, meta_cal=meta_cal)
        result = cp.predict_sets(proba_test, meta_test=meta_test, y_true=y_test)
        
        # Should have coverage per bin
        assert result.per_bin_coverage is not None


class TestDeterminism:
    """Test deterministic behavior with seeds."""
    
    def test_conformal_deterministic_with_seed(self):
        """Conformal prediction should be deterministic."""
        np.random.seed(42)
        y_cal = np.random.randint(0, 2, 100)
        proba_cal = np.random.dirichlet([1, 1], size=100)
        
        y_test = np.random.randint(0, 2, 50)
        proba_test = np.random.dirichlet([1, 1], size=50)
        
        # First run
        cp1 = MondrianConformalClassifier(alpha=0.1)
        cp1.fit(y_cal, proba_cal)
        result1 = cp1.predict_sets(proba_test)
        
        # Second run with same data
        cp2 = MondrianConformalClassifier(alpha=0.1)
        cp2.fit(y_cal, proba_cal)
        result2 = cp2.predict_sets(proba_test)
        
        # Predictions should match
        assert result1.prediction_sets == result2.prediction_sets
        assert result1.set_sizes == result2.set_sizes


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_conformal_invalid_alpha(self):
        """Conformal should reject invalid alpha."""
        with pytest.raises(ValueError, match="alpha"):
            MondrianConformalClassifier(alpha=0.0)
        
        with pytest.raises(ValueError, match="alpha"):
            MondrianConformalClassifier(alpha=1.5)
    
    def test_abstention_invalid_threshold(self):
        """Abstention should reject invalid thresholds."""
        proba = np.array([[0.6, 0.4]])
        y_true = [0]
        
        with pytest.raises(ValueError, match="threshold"):
            evaluate_abstention(proba, y_true, threshold=0.0)
        
        with pytest.raises(ValueError, match="threshold"):
            evaluate_abstention(proba, y_true, threshold=1.5)
    
    def test_calibrator_transform_before_fit(self):
        """Calibrators should require fitting before transform."""
        calibrator = IsotonicCalibrator()
        proba = np.random.rand(10, 2)
        
        with pytest.raises((RuntimeError, AttributeError)):
            calibrator.transform(proba)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_conformal_plus_abstention_workflow(self):
        """Combined conformal + abstention workflow."""
        np.random.seed(42)
        
        # Calibration
        y_cal = np.random.randint(0, 2, 100)
        proba_cal = np.random.dirichlet([1, 1], size=100)
        
        # Test
        y_test = np.random.randint(0, 2, 50)
        proba_test = np.random.dirichlet([1, 1], size=50)
        
        # Conformal prediction
        cp = MondrianConformalClassifier(alpha=0.1)
        cp.fit(y_cal, proba_cal)
        cp_result = cp.predict_sets(proba_test, y_true=y_test)
        
        # Abstention on prediction sets
        abstain_result = evaluate_abstention(
            proba_test,
            y_test,
            threshold=0.7,
            prediction_sets=cp_result.prediction_sets,
            max_set_size=1,
        )
        
        assert abstain_result.abstain_rate >= 0
        assert abstain_result.coverage is not None
        assert len(abstain_result.predictions) == len(y_test)
