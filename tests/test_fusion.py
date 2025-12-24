"""Tests for fusion strategies."""

import numpy as np
import pytest
from foodspec.ml.fusion import (
    late_fusion_concat,
    decision_fusion_vote,
    decision_fusion_weighted,
)


def test_late_fusion_concat():
    """Test feature concatenation across modalities."""
    n_samples = 10
    X_raman = np.random.randn(n_samples, 5)
    X_ftir = np.random.randn(n_samples, 8)
    feature_dict = {"raman": X_raman, "ftir": X_ftir}
    
    result = late_fusion_concat(feature_dict)
    
    assert result.X_fused.shape == (n_samples, 13)
    assert len(result.feature_names) == 13
    assert result.modality_boundaries["raman"] == (0, 5)
    assert result.modality_boundaries["ftir"] == (5, 13)


def test_decision_fusion_vote_majority():
    """Test majority voting fusion."""
    n = 6
    pred1 = np.array([0, 0, 1, 1, 2, 0])
    pred2 = np.array([0, 1, 1, 1, 2, 0])
    pred3 = np.array([0, 0, 1, 2, 2, 1])
    
    result = decision_fusion_vote({"m1": pred1, "m2": pred2, "m3": pred3}, strategy="majority")
    
    # Sample 0: [0,0,0] -> 0
    # Sample 1: [0,1,0] -> 0
    # Sample 2: [1,1,1] -> 1
    assert result.predictions[0] == 0
    assert result.predictions[2] == 1


def test_decision_fusion_vote_unanimous():
    """Test unanimous voting fusion."""
    pred1 = np.array([0, 0, 1, 1])
    pred2 = np.array([0, 1, 1, 2])
    
    result = decision_fusion_vote({"m1": pred1, "m2": pred2}, strategy="unanimous")
    
    # Sample 0: [0,0] -> 0
    # Sample 1: [0,1] -> -1 (no agreement)
    # Sample 2: [1,1] -> 1
    assert result.predictions[0] == 0
    assert result.predictions[1] == -1
    assert result.predictions[2] == 1


def test_decision_fusion_weighted():
    """Test weighted probability averaging."""
    n = 5
    n_classes = 3
    P1 = np.random.rand(n, n_classes)
    P1 /= P1.sum(axis=1, keepdims=True)
    P2 = np.random.rand(n, n_classes)
    P2 /= P2.sum(axis=1, keepdims=True)
    
    result = decision_fusion_weighted({"m1": P1, "m2": P2}, weights={"m1": 0.7, "m2": 0.3})
    
    assert result.probabilities.shape == (n, n_classes)
    assert result.predictions.shape == (n,)
    # Check probabilities sum to 1
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
