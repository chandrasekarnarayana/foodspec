"""Tests for fusion agreement metrics."""

import numpy as np
import pytest
from foodspec.stats.fusion_metrics import (
    modality_agreement_kappa,
    modality_consistency_rate,
    cross_modality_correlation,
)


def test_modality_agreement_kappa():
    """Test Cohen's kappa agreement between modalities."""
    n = 20
    pred1 = np.array([0] * 10 + [1] * 10)
    pred2 = np.array([0] * 9 + [1] * 11)  # mostly agrees
    
    kappa_df = modality_agreement_kappa({"m1": pred1, "m2": pred2})
    
    assert kappa_df.shape == (2, 2)
    assert kappa_df.loc["m1", "m1"] == 1.0
    assert kappa_df.loc["m2", "m2"] == 1.0
    # Cross-modality kappa should be positive but < 1
    assert 0.0 < kappa_df.loc["m1", "m2"] < 1.0


def test_modality_consistency_rate():
    """Test consistency rate calculation."""
    pred1 = np.array([0, 0, 1, 1, 2])
    pred2 = np.array([0, 1, 1, 1, 2])
    # Samples 0,2,4 agree (indices where both are equal)
    rate = modality_consistency_rate({"m1": pred1, "m2": pred2})
    # Sample 0: [0,0] agree, Sample 1: [0,1] no, Sample 2: [1,1] agree
    # Sample 3: [1,1] agree, Sample 4: [2,2] agree
    # So 4 out of 5 agree
    assert np.isclose(rate, 4.0 / 5.0)


def test_cross_modality_correlation():
    """Test cross-modality feature correlation."""
    # Create two modalities with simple positive correlation structure
    n = 50
    # Both modalities measure similar underlying patterns
    X1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]] * 10).astype(float)
    X2 = X1 * 1.2 + 5  # linearly related
    
    corr_df = cross_modality_correlation({"m1": X1, "m2": X2}, method="pearson")
    
    assert corr_df.shape == (2, 2)
    assert corr_df.loc["m1", "m1"] == 1.0
    assert corr_df.loc["m2", "m2"] == 1.0
    # Perfect linear relationship → all pairwise correlations should be 1.0
    assert corr_df.loc["m1", "m2"] > 0.99
    assert corr_df.loc["m2", "m1"] > 0.99
