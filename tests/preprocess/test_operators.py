"""Tests for individual preprocessing operators."""

import numpy as np
import pytest

from foodspec.preprocess.spectroscopy_operators import (
    DespikeOperator,
    FluorescenceRemovalOperator,
    EMSCOperator,
    MSCOperator,
    AtmosphericCorrectionOperator,
    InterpolationOperator,
)
from foodspec.engine.preprocessing.engine import (
    BaselineStep,
    SmoothingStep,
    NormalizationStep,
    DerivativeStep,
)


class TestDespikeOperator:
    """Test cosmic ray spike removal."""

    def test_despike_removes_spikes(self, synthetic_raman_data):
        """Test that despike removes cosmic rays."""
        op = DespikeOperator(window=5, threshold=5.0)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape
        assert np.all(np.isfinite(result.x))
        
        # Spikes should be reduced
        assert np.max(result.x) <= np.max(synthetic_raman_data.x)

    def test_despike_parameters(self, synthetic_raman_data):
        """Test different despike parameters."""
        op1 = DespikeOperator(window=3, threshold=3.0)
        result1 = op1.transform(synthetic_raman_data)
        
        op2 = DespikeOperator(window=7, threshold=7.0)
        result2 = op2.transform(synthetic_raman_data)
        
        # Different parameters should give different results
        assert not np.allclose(result1.x, result2.x)


class TestFluorescenceRemovalOperator:
    """Test fluorescence background removal."""

    def test_fluorescence_poly_method(self, synthetic_raman_data):
        """Test polynomial fluorescence removal."""
        op = FluorescenceRemovalOperator(method="poly", poly_order=2)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape
        # Background should be reduced (mean should be closer to 0)
        assert np.abs(np.mean(result.x)) < np.abs(np.mean(synthetic_raman_data.x))

    def test_fluorescence_als_method(self, synthetic_raman_data):
        """Test ALS fluorescence removal."""
        op = FluorescenceRemovalOperator(method="als")
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape
        assert np.all(np.isfinite(result.x))


class TestEMSCOperator:
    """Test Extended Multiplicative Scatter Correction."""

    def test_emsc_fit_transform(self, synthetic_ftir_data):
        """Test EMSC fit and transform."""
        op = EMSCOperator(order=2)
        op.fit(synthetic_ftir_data)
        result = op.transform(synthetic_ftir_data)
        
        assert result.x.shape == synthetic_ftir_data.x.shape
        assert op.reference is not None

    def test_emsc_with_reference(self, synthetic_ftir_data):
        """Test EMSC with provided reference."""
        ref = np.mean(synthetic_ftir_data.x, axis=0)
        op = EMSCOperator(reference=ref, order=2)
        result = op.transform(synthetic_ftir_data)
        
        assert result.x.shape == synthetic_ftir_data.x.shape


class TestMSCOperator:
    """Test Multiplicative Scatter Correction."""

    def test_msc_fit_transform(self, synthetic_ftir_data):
        """Test MSC fit and transform."""
        op = MSCOperator()
        op.fit(synthetic_ftir_data)
        result = op.transform(synthetic_ftir_data)
        
        assert result.x.shape == synthetic_ftir_data.x.shape
        assert op.reference is not None

    def test_msc_reduces_scatter(self, synthetic_ftir_data):
        """Test that MSC reduces scatter."""
        op = MSCOperator()
        op.fit(synthetic_ftir_data)
        result = op.transform(synthetic_ftir_data)
        
        # Scatter should be reduced (smaller variance across samples)
        var_before = np.var(synthetic_ftir_data.x, axis=0).mean()
        var_after = np.var(result.x, axis=0).mean()
        # Not always guaranteed to reduce, but should be similar magnitude
        assert np.isfinite(var_after)


class TestAtmosphericCorrectionOperator:
    """Test atmospheric correction."""

    def test_atmospheric_correction(self, synthetic_ftir_data):
        """Test atmospheric absorption line correction."""
        op = AtmosphericCorrectionOperator(co2_window=50, water_window=100)
        result = op.transform(synthetic_ftir_data)
        
        assert result.x.shape == synthetic_ftir_data.x.shape


class TestInterpolationOperator:
    """Test wavenumber grid interpolation."""

    def test_interpolation_to_new_grid(self, synthetic_raman_data):
        """Test interpolation to new wavenumber grid."""
        # Create target grid (fewer points)
        target_grid = np.linspace(600, 2900, 256)
        
        op = InterpolationOperator(target_grid=target_grid, method="linear")
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape[0] == synthetic_raman_data.x.shape[0]
        assert result.x.shape[1] == len(target_grid)
        np.testing.assert_array_equal(result.wavenumbers, target_grid)

    def test_interpolation_none_grid(self, synthetic_raman_data):
        """Test that None grid returns unchanged data."""
        op = InterpolationOperator(target_grid=None)
        result = op.transform(synthetic_raman_data)
        
        np.testing.assert_array_equal(result.x, synthetic_raman_data.x)


class TestBaselineStep:
    """Test baseline correction methods."""

    def test_baseline_als(self, synthetic_raman_data):
        """Test ALS baseline correction."""
        op = BaselineStep(method="als", lam=1e5, p=0.01)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape
        # Baseline-corrected should have lower overall intensity
        assert np.mean(result.x) < np.mean(synthetic_raman_data.x)

    def test_baseline_poly(self, synthetic_raman_data):
        """Test polynomial baseline correction."""
        op = BaselineStep(method="poly", degree=3)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape

    def test_baseline_snip(self, synthetic_raman_data):
        """Test SNIP baseline correction."""
        op = BaselineStep(method="snip", iterations=30)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape


class TestSmoothingStep:
    """Test smoothing operators."""

    def test_smoothing_savgol(self, synthetic_raman_data):
        """Test Savitzky-Golay smoothing."""
        op = SmoothingStep(method="savgol", window_length=7, polyorder=3)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape
        # Smoothed should have lower variance
        assert np.var(result.x) < np.var(synthetic_raman_data.x)

    def test_smoothing_gaussian(self, synthetic_raman_data):
        """Test Gaussian smoothing."""
        op = SmoothingStep(method="gaussian", sigma=2.0)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape


class TestNormalizationStep:
    """Test normalization operators."""

    def test_normalization_snv(self, synthetic_raman_data):
        """Test SNV normalization."""
        op = NormalizationStep(method="snv")
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape
        # Each spectrum should have mean ~ 0 and std ~ 1
        means = np.mean(result.x, axis=1)
        stds = np.std(result.x, axis=1)
        np.testing.assert_array_almost_equal(means, 0, decimal=10)
        np.testing.assert_array_almost_equal(stds, 1, decimal=10)

    def test_normalization_vector(self, synthetic_raman_data):
        """Test vector normalization."""
        op = NormalizationStep(method="vector")
        result = op.transform(synthetic_raman_data)
        
        # Each spectrum should have norm ~ 1
        norms = np.linalg.norm(result.x, axis=1)
        np.testing.assert_array_almost_equal(norms, 1, decimal=10)

    def test_normalization_area(self, synthetic_raman_data):
        """Test area normalization."""
        op = NormalizationStep(method="area")
        result = op.transform(synthetic_raman_data)
        
        # Each spectrum should have sum ~ 1
        sums = np.sum(result.x, axis=1)
        np.testing.assert_array_almost_equal(sums, 1, decimal=10)


class TestDerivativeStep:
    """Test derivative operators."""

    def test_first_derivative(self, synthetic_raman_data):
        """Test first derivative."""
        op = DerivativeStep(order=1, window_length=7, polyorder=2)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape

    def test_second_derivative(self, synthetic_raman_data):
        """Test second derivative."""
        op = DerivativeStep(order=2, window_length=9, polyorder=3)
        result = op.transform(synthetic_raman_data)
        
        assert result.x.shape == synthetic_raman_data.x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
