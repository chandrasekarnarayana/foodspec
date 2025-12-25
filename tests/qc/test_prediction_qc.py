"""
Tests for QC prediction quality control module.
"""
import numpy as np
import pytest

from foodspec.qc.prediction_qc import (
    evaluate_prediction_qc,
    PredictionQCResult
)


def test_evaluate_prediction_qc_all_pass():
    """Test full QC evaluation with all checks passing."""
    probs = np.array([0.85, 0.10, 0.05])
    
    result = evaluate_prediction_qc(
        probs=probs,
        drift_score=0.05,
        ece=0.04,
        min_confidence=0.6,
        drift_threshold=0.2,
        ece_threshold=0.08
    )
    
    assert isinstance(result, PredictionQCResult)
    assert result.do_not_trust is False


def test_evaluate_prediction_qc_low_confidence():
    """Test QC evaluation with low confidence."""
    probs = np.array([0.45, 0.35, 0.20])
    
    result = evaluate_prediction_qc(
        probs=probs,
        drift_score=0.05,
        ece=0.04,
        min_confidence=0.6,
        drift_threshold=0.2,
        ece_threshold=0.08
    )
    
    assert result.do_not_trust is True


def test_evaluate_prediction_qc_drift_warning():
    """Test QC evaluation with drift warning."""
    probs = np.array([0.75, 0.15, 0.10])
    
    result = evaluate_prediction_qc(
        probs=probs,
        drift_score=0.25,
        ece=0.04,
        min_confidence=0.6,
        drift_threshold=0.2,
        ece_threshold=0.08
    )
    
    assert result.do_not_trust is True
    assert any('drift' in r.lower() for r in result.reasons)


def test_evaluate_prediction_qc_poor_calibration():
    """Test QC evaluation with poor calibration."""
    probs = np.array([0.75, 0.15, 0.10])
    
    result = evaluate_prediction_qc(
        probs=probs,
        drift_score=0.05,
        ece=0.15,
        min_confidence=0.6,
        drift_threshold=0.2,
        ece_threshold=0.08
    )
    
    assert result.do_not_trust is True
    assert any('calibration' in r.lower() or 'ece' in r.lower() for r in result.reasons)


def test_evaluate_prediction_qc_multiple_issues():
    """Test QC evaluation with multiple issues."""
    probs = np.array([0.50, 0.30, 0.20])
    
    result = evaluate_prediction_qc(
        probs=probs,
        drift_score=0.25,
        ece=0.12,
        min_confidence=0.6,
        drift_threshold=0.2,
        ece_threshold=0.08
    )
    
    assert result.do_not_trust is True
    assert len(result.reasons) >= 1


def test_prediction_qc_result_summary():
    """Test PredictionQCResult summary method."""
    result = PredictionQCResult(
        do_not_trust=True,
        warnings=['test warning'],
        reasons=['test reason']
    )
    
    summary = result.summary()
    assert 'Do not trust' in summary or 'trust' in summary.lower()


def test_evaluate_prediction_qc_with_list_input():
    """Test QC evaluation with list input instead of numpy array."""
    probs = [0.85, 0.10, 0.05]
    
    result = evaluate_prediction_qc(
        probs=probs,
        drift_score=0.05,
        ece=0.04,
        min_confidence=0.6
    )
    
    assert isinstance(result, PredictionQCResult)


def test_evaluate_prediction_qc_no_drift_ece():
    """Test QC evaluation with only confidence check."""
    probs = np.array([0.85, 0.10, 0.05])
    
    result = evaluate_prediction_qc(
        probs=probs,
        drift_score=None,
        ece=None,
        min_confidence=0.6
    )
    
    assert isinstance(result, PredictionQCResult)
    assert result.do_not_trust is False
