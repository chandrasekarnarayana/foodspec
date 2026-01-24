"""
Phase 3 comprehensive tests for BandIntegration feature extractor.

Tests trapezoidal integration and mean methods on known functions
where analytic values can be computed exactly.
"""

import numpy as np
import pytest

from foodspec.features.bands import BandIntegration


# ============================================================================
# Known Function Fixtures
# ============================================================================


@pytest.fixture
def linear_spectrum():
    """Linear spectrum y = 2*x + 1 where integral can be computed analytically.
    
    For a linear function y = mx + b over [a, b]:
    Integral = (m/2)(b² - a²) + b(b - a)
    
    For y = 2x + 1 over [1000, 1500]:
    Integral = (2/2)(1500² - 1000²) + 1(1500 - 1000)
            = (2250000 - 1000000) + 500
            = 1250500
    """
    x = np.linspace(1000, 2000, 1001)  # 1 cm^-1 spacing
    y = 2 * x + 1  # Linear function
    
    X = np.vstack([y, y * 2, y * 0.5])  # Three scaled versions
    
    return X, x


@pytest.fixture
def constant_spectrum():
    """Constant spectrum y = 5.0 for easy integral calculation.
    
    Integral of constant c over [a, b] = c * (b - a)
    """
    x = np.linspace(1000, 2000, 1001)
    y = np.full_like(x, 5.0)
    
    X = y.reshape(1, -1)
    
    return X, x


@pytest.fixture
def spectrum_with_baseline():
    """Spectrum with added linear baseline for testing baseline correction."""
    x = np.linspace(1000, 2000, 1001)
    
    # Peak without baseline
    peak = 10 * np.exp(-0.5 * ((x - 1500) / 50) ** 2)
    
    # Linear baseline
    baseline = 0.002 * x + 3.0
    
    # Create two samples: with and without baseline
    X = np.vstack([
        peak,
        peak + baseline
    ])
    
    return X, x


# ============================================================================
# BandIntegration Tests - Analytic Verification
# ============================================================================


class TestBandIntegrationPhase3:
    """Comprehensive tests for BandIntegration with analytic verification."""
    
    def test_trapz_linear_function_exact(self, linear_spectrum):
        """Acceptance test: Integration matches analytic values for linear spectrum.
        
        For y = 2x + 1 over [1000, 1500]:
        Analytic integral = 1,250,500
        """
        X, x = linear_spectrum
        
        extractor = BandIntegration(
            bands=[(1000, 1500)],
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # Expected analytic value
        # Integral of (2x + 1) from 1000 to 1500:
        # = [x² + x] from 1000 to 1500
        # = (1500² + 1500) - (1000² + 1000)
        # = 2251500 - 1001000 = 1250500
        expected = 1250500.0
        
        # Check first sample (tolerance for numerical integration)
        assert np.abs(features.iloc[0, 0] - expected) < expected * 0.001  # Within 0.1%
        
        # Second sample should be 2x
        assert np.abs(features.iloc[1, 0] - expected * 2) < expected * 0.002
        
        # Third sample should be 0.5x
        assert np.abs(features.iloc[2, 0] - expected * 0.5) < expected * 0.001
    
    def test_trapz_constant_function_exact(self, constant_spectrum):
        """Test integration of constant function with exact analytic result."""
        X, x = constant_spectrum
        
        extractor = BandIntegration(
            bands=[(1200, 1300), (1600, 1800)],
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # For constant y = 5.0:
        # Band 1: [1200, 1300] → integral = 5.0 * (1300 - 1200) = 500.0
        # Band 2: [1600, 1800] → integral = 5.0 * (1800 - 1600) = 1000.0
        
        assert np.abs(features.iloc[0, 0] - 500.0) < 0.1
        assert np.abs(features.iloc[0, 1] - 1000.0) < 0.1
    
    def test_trapz_multiple_bands(self, linear_spectrum):
        """Test multiple bands with known integrals."""
        X, x = linear_spectrum
        
        extractor = BandIntegration(
            bands=[(1000, 1200), (1400, 1600), (1800, 2000)],
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # Check shape
        assert features.shape == (3, 3)  # 3 samples, 3 bands
        
        # For y = 2x + 1:
        # Band 1: [1000, 1200] → integral = [x² + x]₁₀₀₀¹²⁰⁰ = 441200 - 1001000 = -559800 ??? 
        # Wait, that's wrong. Let me recalculate:
        # Integral of (2x + 1) from a to b = [x² + x]_a^b = (b² + b) - (a² + a)
        # [1000, 1200]: (1200² + 1200) - (1000² + 1000) = 1441200 - 1001000 = 440200
        
        # Verify first band approximately
        band1_expected = 440200.0
        assert np.abs(features.iloc[0, 0] - band1_expected) < band1_expected * 0.001
    
    def test_mean_method(self, linear_spectrum):
        """Test mean method computes average intensity."""
        X, x = linear_spectrum
        
        extractor = BandIntegration(
            bands=[(1000, 1500)],
            method="mean",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # For linear y = 2x + 1 over [1000, 1500]:
        # Mean = (y(1000) + y(1500)) / 2 = ((2*1000+1) + (2*1500+1)) / 2
        #      = (2001 + 3001) / 2 = 2501
        
        # With uniform grid, mean should be close to midpoint value
        expected_mean = 2501.0
        assert np.abs(features.iloc[0, 0] - expected_mean) < 10.0
    
    def test_baseline_correction_linear(self, spectrum_with_baseline):
        """Test linear baseline correction removes baseline effectively."""
        X, x = spectrum_with_baseline
        
        # With baseline correction
        extractor_corrected = BandIntegration(
            bands=[(1450, 1550)],
            method="trapz",
            baseline="linear"
        )
        
        features_corrected = extractor_corrected.compute(X, x)
        
        # Without baseline correction
        extractor_uncorrected = BandIntegration(
            bands=[(1450, 1550)],
            method="trapz",
            baseline="none"
        )
        
        features_uncorrected = extractor_uncorrected.compute(X, x)
        
        # Sample 0 (no baseline) and sample 1 (with baseline) should be similar after correction
        diff_corrected = np.abs(features_corrected.iloc[1, 0] - features_corrected.iloc[0, 0])
        diff_uncorrected = np.abs(features_uncorrected.iloc[1, 0] - features_uncorrected.iloc[0, 0])
        
        # Correction should reduce difference by at least 50%
        assert diff_corrected < diff_uncorrected * 0.5
    
    def test_feature_names_trapz(self, linear_spectrum):
        """Test feature naming with trapz method."""
        X, x = linear_spectrum
        
        extractor = BandIntegration(
            bands=[(1000, 1200), (1400, 1600, "amide_I")],
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        assert list(features.columns) == ["band_trapz@1200-1000", "band_trapz@amide_I"]
    
    def test_feature_names_mean(self, linear_spectrum):
        """Test feature naming with mean method."""
        X, x = linear_spectrum
        
        extractor = BandIntegration(
            bands=[(1000, 1200), (1400, 1600)],
            method="mean",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        assert list(features.columns) == ["band_mean@1200-1000", "band_mean@1600-1400"]
    
    def test_parameter_validation(self):
        """Test parameter validation in __post_init__."""
        # Empty bands
        with pytest.raises(ValueError, match="non-empty"):
            BandIntegration(bands=[])
        
        # Invalid method
        with pytest.raises(ValueError, match="'trapz' or 'mean'"):
            BandIntegration(bands=[(1000, 1500)], method="simpson")
        
        # Invalid baseline
        with pytest.raises(ValueError, match="'none' or 'linear'"):
            BandIntegration(bands=[(1000, 1500)], baseline="polynomial")
        
        # Band with < 2 elements
        with pytest.raises(ValueError, match="at least 2 elements"):
            BandIntegration(bands=[(1000,)])
        
        # NaN in band
        with pytest.raises(ValueError, match="NaN"):
            BandIntegration(bands=[(1000, np.nan)])
        
        # Start >= end
        with pytest.raises(ValueError, match="start must be < end"):
            BandIntegration(bands=[(1500, 1000)])
    
    def test_input_validation(self, linear_spectrum):
        """Test input validation in compute()."""
        X, x = linear_spectrum
        extractor = BandIntegration(bands=[(1000, 1500)])
        
        # X not 2D
        with pytest.raises(ValueError, match="2D"):
            extractor.compute(x, x)
        
        # x doesn't match X columns
        with pytest.raises(ValueError, match="match X columns"):
            extractor.compute(X, x[:500])
        
        # NaN in X
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extractor.compute(X_nan, x)
        
        # NaN in x
        x_nan = x.copy()
        x_nan[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extractor.compute(X, x_nan)


# ============================================================================
# Determinism Tests
# ============================================================================


class TestBandDeterminism:
    """Test that BandIntegration produces deterministic results."""
    
    def test_trapz_deterministic(self, linear_spectrum):
        """Test trapz gives identical results on repeated calls."""
        X, x = linear_spectrum
        
        extractor = BandIntegration(
            bands=[(1000, 1500), (1600, 1800)],
            method="trapz",
            baseline="none"
        )
        
        results1 = extractor.compute(X, x)
        results2 = extractor.compute(X, x)
        
        assert np.array_equal(results1.values, results2.values)
    
    def test_mean_deterministic(self, linear_spectrum):
        """Test mean gives identical results on repeated calls."""
        X, x = linear_spectrum
        
        extractor = BandIntegration(
            bands=[(1000, 1500)],
            method="mean",
            baseline="linear"
        )
        
        results1 = extractor.compute(X, x)
        results2 = extractor.compute(X, x)
        
        assert np.array_equal(results1.values, results2.values)


# ============================================================================
# Edge Cases
# ============================================================================


class TestBandEdgeCases:
    """Test edge cases and robustness."""
    
    def test_band_at_grid_boundary(self):
        """Test band extraction when band is at edge of grid."""
        x = np.linspace(1000, 2000, 1001)
        y = 2 * x + 1
        X = y.reshape(1, -1)
        
        extractor = BandIntegration(
            bands=[(1000, 1050)],  # At start of grid
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # Should handle boundary gracefully
        assert features.iloc[0, 0] > 0
    
    def test_very_narrow_band(self):
        """Test integration for very narrow bands."""
        x = np.linspace(1000, 2000, 2001)  # High resolution (0.5 cm^-1)
        y = 2 * x + 1
        X = y.reshape(1, -1)
        
        # Band only 2 cm^-1 wide
        extractor = BandIntegration(
            bands=[(1499, 1501)],
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # For y = 2x + 1 over [1499, 1501]:
        # Integral ≈ 2*1500*2 + 1*2 = 6002
        expected = 6002.0
        assert np.abs(features.iloc[0, 0] - expected) < 10.0
    
    def test_overlapping_bands(self):
        """Test multiple overlapping bands."""
        x = np.linspace(1000, 2000, 1001)
        y = 2 * x + 1
        X = y.reshape(1, -1)
        
        extractor = BandIntegration(
            bands=[(1000, 1500), (1400, 1600)],  # Overlapping
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # Both bands should have positive integrals
        assert features.iloc[0, 0] > 0
        assert features.iloc[0, 1] > 0
    
    def test_single_point_band(self):
        """Test band with only one point (edge case)."""
        x = np.linspace(1000, 2000, 11)  # Very coarse: 100 cm^-1 spacing
        y = 2 * x + 1
        X = y.reshape(1, -1)
        
        # Band that might contain only one point
        extractor = BandIntegration(
            bands=[(1050, 1150)],  # Between grid points
            method="mean",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # Should still return a value (nearest point)
        assert np.isfinite(features.iloc[0, 0])


# ============================================================================
# Comparison with Expected Values
# ============================================================================


class TestAnalyticComparison:
    """Compare integration results with expected analytic values."""
    
    def test_quadratic_function(self):
        """Test integration of quadratic function y = x².
        
        Integral of x² from a to b = (b³ - a³) / 3
        """
        x = np.linspace(0, 100, 1001)
        y = x ** 2
        X = y.reshape(1, -1)
        
        extractor = BandIntegration(
            bands=[(10, 20), (30, 50)],
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # Band 1: [10, 20] → (20³ - 10³) / 3 = (8000 - 1000) / 3 = 2333.33
        # Band 2: [30, 50] → (50³ - 30³) / 3 = (125000 - 27000) / 3 = 32666.67
        
        assert np.abs(features.iloc[0, 0] - 2333.33) < 10.0
        assert np.abs(features.iloc[0, 1] - 32666.67) < 50.0
    
    def test_exponential_decay(self):
        """Test integration of exponential decay."""
        x = np.linspace(0, 10, 1001)
        y = np.exp(-x)
        X = y.reshape(1, -1)
        
        extractor = BandIntegration(
            bands=[(0, 5)],
            method="trapz",
            baseline="none"
        )
        
        features = extractor.compute(X, x)
        
        # Integral of e^(-x) from 0 to 5 = [-e^(-x)]₀⁵ = -e^(-5) - (-e^0)
        #                                   = 1 - e^(-5) ≈ 0.9933
        expected = 1 - np.exp(-5)
        
        assert np.abs(features.iloc[0, 0] - expected) < 0.01
