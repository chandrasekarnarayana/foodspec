"""
Phase 2 comprehensive tests for peak-based feature extractors.

Tests peak heights, areas, and ratios with controlled synthetic spectra
where peak positions and magnitudes are known exactly.
"""

import numpy as np
import pytest

from foodspec.features.peaks import PeakAreas, PeakHeights, PeakRatios


# ============================================================================
# Synthetic Spectrum Fixtures
# ============================================================================


def gaussian_peak(x, center, amplitude, sigma=5.0):
    """Create Gaussian peak centered at `center` with given amplitude."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


@pytest.fixture
def synthetic_spectrum_known_peaks():
    """Synthetic spectrum with 3 known Gaussian peaks.
    
    Peaks:
    - Peak 1: center=1450 cm^-1, amplitude=10.0
    - Peak 2: center=1650 cm^-1, amplitude=5.0
    - Peak 3: center=1750 cm^-1, amplitude=8.0
    """
    x = np.linspace(1400, 1800, 401)  # 1 cm^-1 resolution
    y1 = gaussian_peak(x, 1450, 10.0, sigma=10.0)
    y2 = gaussian_peak(x, 1650, 5.0, sigma=8.0)
    y3 = gaussian_peak(x, 1750, 8.0, sigma=6.0)
    
    spectrum = y1 + y2 + y3
    
    # Create 3 samples with different scaling
    X = np.vstack([
        spectrum,           # Sample 0: original
        spectrum * 2.0,     # Sample 1: 2x magnitude
        spectrum * 0.5,     # Sample 2: 0.5x magnitude
    ])
    
    return X, x


@pytest.fixture
def synthetic_spectrum_baseline():
    """Synthetic spectrum with baseline offset for testing baseline correction."""
    x = np.linspace(1000, 2000, 501)
    
    # Peak at 1500 with linear baseline
    peak = gaussian_peak(x, 1500, 20.0, sigma=15.0)
    baseline = 0.01 * x + 5.0  # Linear baseline
    
    spectrum_no_baseline = peak
    spectrum_with_baseline = peak + baseline
    
    X = np.vstack([
        spectrum_no_baseline,
        spectrum_with_baseline,
    ])
    
    return X, x


# ============================================================================
# PeakHeights Tests
# ============================================================================


class TestPeakHeightsPhase2:
    """Comprehensive tests for PeakHeights extractor."""
    
    def test_heights_exact_values(self, synthetic_spectrum_known_peaks):
        """Test that peak heights match known amplitudes within tolerance."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor = PeakHeights(peaks=[1450, 1650, 1750], window=15.0, method="max")
        features = extractor.compute(X, x)
        
        # Check shape and column names
        assert features.shape == (3, 3)
        assert list(features.columns) == ["height@1450", "height@1650", "height@1750"]
        
        # Sample 0: check absolute values (within 1% tolerance)
        assert np.abs(features.iloc[0, 0] - 10.0) < 0.1  # Peak 1
        assert np.abs(features.iloc[0, 1] - 5.0) < 0.05   # Peak 2
        assert np.abs(features.iloc[0, 2] - 8.0) < 0.08   # Peak 3
        
        # Samples 1 and 2: check scaling relationships
        assert np.allclose(features.iloc[1, :], features.iloc[0, :] * 2.0, rtol=0.01)
        assert np.allclose(features.iloc[2, :], features.iloc[0, :] * 0.5, rtol=0.01)
    
    def test_heights_method_max_vs_mean(self, synthetic_spectrum_known_peaks):
        """Test difference between max and mean methods."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor_max = PeakHeights(peaks=[1650], window=20.0, method="max")
        extractor_mean = PeakHeights(peaks=[1650], window=20.0, method="mean")
        
        features_max = extractor_max.compute(X, x)
        features_mean = extractor_mean.compute(X, x)
        
        # Max should be higher than mean for Gaussian peaks
        assert features_max.iloc[0, 0] > features_mean.iloc[0, 0]
        
        # Mean should be roughly 0.4-0.6 of max for Gaussian with this window (larger window = lower ratio)
        ratio = features_mean.iloc[0, 0] / features_max.iloc[0, 0]
        assert 0.4 < ratio < 0.7
    
    def test_heights_window_none(self, synthetic_spectrum_known_peaks):
        """Test nearest-point mode with window=None."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor = PeakHeights(peaks=[1650], window=None, method="max")
        features = extractor.compute(X, x)
        
        # Should still capture peak reasonably (exact center)
        assert features.iloc[0, 0] > 4.5  # Near amplitude of 5.0
    
    def test_heights_parameter_validation(self):
        """Test parameter validation in __post_init__."""
        # Empty peaks
        with pytest.raises(ValueError, match="non-empty"):
            PeakHeights(peaks=[])
        
        # Invalid method
        with pytest.raises(ValueError, match="'max' or 'mean'"):
            PeakHeights(peaks=[1500], method="median")
        
        # Negative window
        with pytest.raises(ValueError, match="positive"):
            PeakHeights(peaks=[1500], window=-5.0)
        
        # NaN in peaks
        with pytest.raises(ValueError, match="NaN"):
            PeakHeights(peaks=[1500, np.nan])
    
    def test_heights_input_validation(self, synthetic_spectrum_known_peaks):
        """Test input validation in compute()."""
        X, x = synthetic_spectrum_known_peaks
        extractor = PeakHeights(peaks=[1500])
        
        # X not 2D
        with pytest.raises(ValueError, match="2D"):
            extractor.compute(x, x)  # 1D input
        
        # x doesn't match X columns
        with pytest.raises(ValueError, match="match X columns"):
            extractor.compute(X, x[:300])
        
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
# PeakAreas Tests
# ============================================================================


class TestPeakAreasPhase2:
    """Comprehensive tests for PeakAreas extractor."""
    
    def test_areas_increase_with_magnitude(self, synthetic_spectrum_known_peaks):
        """Acceptance test: area integration increases with signal magnitude."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor = PeakAreas(bands=[(1440, 1460), (1640, 1660), (1740, 1760)], baseline="none")
        features = extractor.compute(X, x)
        
        # Check shape and column names
        assert features.shape == (3, 3)
        assert "area@1460-1440" in features.columns[0]
        
        # Sample 1 should have 2x area of Sample 0
        assert np.allclose(features.iloc[1, :], features.iloc[0, :] * 2.0, rtol=0.02)
        
        # Sample 2 should have 0.5x area of Sample 0
        assert np.allclose(features.iloc[2, :], features.iloc[0, :] * 0.5, rtol=0.02)
    
    def test_areas_baseline_correction(self, synthetic_spectrum_baseline):
        """Test baseline='linear' removes linear baseline."""
        X, x = synthetic_spectrum_baseline
        
        # With baseline correction
        extractor_corrected = PeakAreas(bands=[(1470, 1530)], baseline="linear")
        features_corrected = extractor_corrected.compute(X, x)
        
        # Without baseline correction
        extractor_uncorrected = PeakAreas(bands=[(1470, 1530)], baseline="none")
        features_uncorrected = extractor_uncorrected.compute(X, x)
        
        # Sample 0 (no baseline): both methods should give similar results (within 30% due to edge effects)
        ratio = features_corrected.iloc[0, 0] / features_uncorrected.iloc[0, 0]
        assert 0.7 < ratio < 1.3
        
        # Sample 1 (with baseline): corrected should be closer to sample 0
        diff_corrected = np.abs(features_corrected.iloc[1, 0] - features_corrected.iloc[0, 0])
        diff_uncorrected = np.abs(features_uncorrected.iloc[1, 0] - features_uncorrected.iloc[0, 0])
        
        # Correction should significantly reduce difference (at least 50% improvement)
        assert diff_corrected < diff_uncorrected * 0.5
    
    def test_areas_band_labels(self, synthetic_spectrum_known_peaks):
        """Test custom band labels in column names."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor = PeakAreas(
            bands=[(1440, 1460, "amide_II"), (1640, 1660, "amide_I")],
            baseline="linear"
        )
        features = extractor.compute(X, x)
        
        assert list(features.columns) == ["area@amide_II", "area@amide_I"]
    
    def test_areas_parameter_validation(self):
        """Test parameter validation in __post_init__."""
        # Empty bands
        with pytest.raises(ValueError, match="non-empty"):
            PeakAreas(bands=[])
        
        # Invalid baseline
        with pytest.raises(ValueError, match="'none' or 'linear'"):
            PeakAreas(bands=[(1400, 1500)], baseline="quadratic")
        
        # Band with < 2 elements
        with pytest.raises(ValueError, match="at least 2 elements"):
            PeakAreas(bands=[(1400,)])
        
        # NaN in band
        with pytest.raises(ValueError, match="NaN"):
            PeakAreas(bands=[(1400, np.nan)])
        
        # Low >= high
        with pytest.raises(ValueError, match="low must be < high"):
            PeakAreas(bands=[(1500, 1400)])
    
    def test_areas_input_validation(self, synthetic_spectrum_known_peaks):
        """Test input validation in compute()."""
        X, x = synthetic_spectrum_known_peaks
        extractor = PeakAreas(bands=[(1400, 1500)])
        
        # X not 2D
        with pytest.raises(ValueError, match="2D"):
            extractor.compute(x, x)
        
        # x doesn't match X columns
        with pytest.raises(ValueError, match="match X columns"):
            extractor.compute(X, x[:300])
        
        # NaN in X
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extractor.compute(X_nan, x)


# ============================================================================
# PeakRatios Tests
# ============================================================================


class TestPeakRatiosPhase2:
    """Comprehensive tests for PeakRatios extractor."""
    
    def test_ratios_correct_for_synthetic(self, synthetic_spectrum_known_peaks):
        """Acceptance test: ratio features are correct for controlled synthetic spectrum."""
        X, x = synthetic_spectrum_known_peaks
        
        # Known amplitudes: 1450=10.0, 1650=5.0, 1750=8.0
        # Expected ratios: 1450/1650 ≈ 2.0, 1650/1750 ≈ 0.625
        
        extractor = PeakRatios(pairs=[(1450, 1650), (1650, 1750)], method="height", window=15.0)
        features = extractor.compute(X, x)
        
        # Check shape and column names
        assert features.shape == (3, 2)
        assert list(features.columns) == ["ratio@1450/1650", "ratio@1650/1750"]
        
        # Sample 0: check ratios (within 5% tolerance)
        assert np.abs(features.iloc[0, 0] - 2.0) < 0.1      # 10/5 = 2.0
        assert np.abs(features.iloc[0, 1] - 0.625) < 0.05   # 5/8 = 0.625
        
        # Ratios should be consistent across scaled samples
        assert np.allclose(features.iloc[0, :], features.iloc[1, :], rtol=0.01)
        assert np.allclose(features.iloc[0, :], features.iloc[2, :], rtol=0.01)
    
    def test_ratios_method_height_vs_area(self, synthetic_spectrum_known_peaks):
        """Test height-based vs area-based ratios."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor_height = PeakRatios(pairs=[(1450, 1650)], method="height", window=20.0)
        extractor_area = PeakRatios(pairs=[(1450, 1650)], method="area", window=20.0)
        
        features_height = extractor_height.compute(X, x)
        features_area = extractor_area.compute(X, x)
        
        # Both should be positive
        assert features_height.iloc[0, 0] > 0
        assert features_area.iloc[0, 0] > 0
        
        # For Gaussian peaks with different widths, ratios might differ slightly
        # but should be in same ballpark (within 20%)
        ratio_diff = np.abs(features_height.iloc[0, 0] - features_area.iloc[0, 0])
        assert ratio_diff < 0.5  # Should be reasonably close
    
    def test_ratios_window_none(self, synthetic_spectrum_known_peaks):
        """Test nearest-point mode with window=None."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor = PeakRatios(pairs=[(1450, 1650)], method="height", window=None)
        features = extractor.compute(X, x)
        
        # Should still give reasonable ratio
        assert 1.5 < features.iloc[0, 0] < 2.5
    
    def test_ratios_eps_prevents_division_by_zero(self):
        """Test that eps parameter prevents division by zero."""
        x = np.linspace(1000, 2000, 501)
        peak = gaussian_peak(x, 1500, 5.0)
        flat = np.zeros_like(x)
        
        # Create X with two samples: one with peak, one flat
        X = np.vstack([peak, flat])
        
        # This should not raise division by zero (denominator is at 1800 where signal is ~0)
        extractor = PeakRatios(pairs=[(1500, 1800)], method="height", window=10.0, eps=1e-12)
        features = extractor.compute(X, x)
        
        assert np.isfinite(features.iloc[0, 0])
        assert np.isfinite(features.iloc[1, 0])
    
    def test_ratios_parameter_validation(self):
        """Test parameter validation in __post_init__."""
        # Empty pairs
        with pytest.raises(ValueError, match="non-empty"):
            PeakRatios(pairs=[])
        
        # Invalid method
        with pytest.raises(ValueError, match="'height' or 'area'"):
            PeakRatios(pairs=[(1400, 1500)], method="volume")
        
        # Negative window
        with pytest.raises(ValueError, match="positive"):
            PeakRatios(pairs=[(1400, 1500)], window=-5.0)
        
        # Pair with wrong number of elements
        with pytest.raises(ValueError, match="exactly 2 elements"):
            PeakRatios(pairs=[(1400, 1500, 1600)])
        
        # NaN in pair
        with pytest.raises(ValueError, match="NaN"):
            PeakRatios(pairs=[(1400, np.nan)])
    
    def test_ratios_input_validation(self, synthetic_spectrum_known_peaks):
        """Test input validation in compute()."""
        X, x = synthetic_spectrum_known_peaks
        extractor = PeakRatios(pairs=[(1450, 1650)])
        
        # X not 2D
        with pytest.raises(ValueError, match="2D"):
            extractor.compute(x, x)
        
        # x doesn't match X columns
        with pytest.raises(ValueError, match="match X columns"):
            extractor.compute(X, x[:300])
        
        # NaN in X
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extractor.compute(X_nan, x)


# ============================================================================
# Determinism Tests
# ============================================================================


class TestPeakDeterminism:
    """Test that peak extractors produce deterministic results."""
    
    def test_heights_deterministic(self, synthetic_spectrum_known_peaks):
        """Test PeakHeights gives identical results on repeated calls."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor = PeakHeights(peaks=[1450, 1650], window=10.0, method="max")
        
        results1 = extractor.compute(X, x)
        results2 = extractor.compute(X, x)
        
        assert np.array_equal(results1.values, results2.values)
    
    def test_areas_deterministic(self, synthetic_spectrum_known_peaks):
        """Test PeakAreas gives identical results on repeated calls."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor = PeakAreas(bands=[(1440, 1460), (1640, 1660)], baseline="linear")
        
        results1 = extractor.compute(X, x)
        results2 = extractor.compute(X, x)
        
        assert np.array_equal(results1.values, results2.values)
    
    def test_ratios_deterministic(self, synthetic_spectrum_known_peaks):
        """Test PeakRatios gives identical results on repeated calls."""
        X, x = synthetic_spectrum_known_peaks
        
        extractor = PeakRatios(pairs=[(1450, 1650)], method="height", window=10.0)
        
        results1 = extractor.compute(X, x)
        results2 = extractor.compute(X, x)
        
        assert np.array_equal(results1.values, results2.values)


# ============================================================================
# Edge Cases
# ============================================================================


class TestPeakEdgeCases:
    """Test edge cases and robustness."""
    
    def test_peak_at_grid_boundary(self):
        """Test peak extraction when peak is at edge of grid."""
        x = np.linspace(1000, 2000, 501)
        peak_at_edge = gaussian_peak(x, 1000, 10.0, sigma=5.0)
        X = peak_at_edge.reshape(1, -1)
        
        extractor = PeakHeights(peaks=[1000], window=10.0, method="max")
        features = extractor.compute(X, x)
        
        # Should handle boundary gracefully
        assert features.iloc[0, 0] > 0
    
    def test_very_narrow_band(self):
        """Test area extraction for very narrow bands."""
        x = np.linspace(1000, 2000, 1001)  # High resolution
        spectrum = gaussian_peak(x, 1500, 10.0, sigma=2.0)
        X = spectrum.reshape(1, -1)
        
        # Band only 2 cm^-1 wide
        extractor = PeakAreas(bands=[(1499, 1501)], baseline="none")
        features = extractor.compute(X, x)
        
        assert features.iloc[0, 0] > 0  # Should capture some area
    
    def test_overlapping_peaks(self):
        """Test extractors with overlapping peaks."""
        x = np.linspace(1000, 2000, 501)
        y = gaussian_peak(x, 1500, 10.0, sigma=20.0) + gaussian_peak(x, 1550, 8.0, sigma=20.0)
        X = y.reshape(1, -1)
        
        # Extract both peaks
        extractor = PeakHeights(peaks=[1500, 1550], window=30.0, method="max")
        features = extractor.compute(X, x)
        
        # Both should have positive heights
        assert features.iloc[0, 0] > 0
        assert features.iloc[0, 1] > 0
