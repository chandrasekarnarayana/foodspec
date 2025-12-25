"""Comprehensive tests for preprocessing module coverage.

Tests for baseline correction, normalization, smoothing, spike removal,
cropping, and FTIR/Raman specific preprocessing.
"""

import numpy as np
import pytest

from foodspec.preprocess.baseline import ALSBaseline, RubberbandBaseline, PolynomialBaseline
from foodspec.preprocess.normalization import (
    VectorNormalizer, AreaNormalizer, InternalPeakNormalizer,
    SNVNormalizer, MSCNormalizer
)
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother, MovingAverageSmoother
from foodspec.preprocess.spikes import correct_cosmic_rays
from foodspec.preprocess.cropping import RangeCropper


class TestALSBaseline:
    """Tests for Asymmetric Least Squares baseline correction."""

    def test_als_init_default(self):
        """Test ALS initialization with defaults."""
        als = ALSBaseline()
        assert als.lambda_ == 1e5
        assert als.p == 0.001
        assert als.max_iter == 10

    def test_als_init_custom(self):
        """Test ALS initialization with custom parameters."""
        als = ALSBaseline(lambda_=1e4, p=0.01, max_iter=5)
        assert als.lambda_ == 1e4
        assert als.p == 0.01
        assert als.max_iter == 5

    def test_als_fit(self):
        """Test ALS fit method."""
        X = np.random.randn(10, 100)
        als = ALSBaseline()
        result = als.fit(X)
        assert result is als

    def test_als_transform_basic(self):
        """Test ALS transform on simple data."""
        X = np.random.randn(5, 50) + 2.0  # Add baseline
        als = ALSBaseline(lambda_=1e4, p=0.01, max_iter=3)
        Y = als.transform(X)
        assert Y.shape == X.shape
        assert isinstance(Y, np.ndarray)

    def test_als_transform_baseline_removal(self):
        """Test that ALS removes baseline from data."""
        # Create data with clear baseline
        x = np.linspace(0, 10, 100)
        baseline = 5.0 + 2.0 * x / 10
        signal = 1.0 * np.sin(x)
        X = (baseline + signal).reshape(1, -1)
        
        als = ALSBaseline(lambda_=1e4, p=0.01, max_iter=5)
        Y = als.transform(X)
        
        # Corrected signal should have lower mean than original
        assert Y.mean() < X.mean()

    def test_als_invalid_lambda(self):
        """Test ALS with invalid lambda."""
        X = np.random.randn(5, 50)
        als = ALSBaseline(lambda_=-1)
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            als.transform(X)

    def test_als_invalid_p(self):
        """Test ALS with invalid p value."""
        X = np.random.randn(5, 50)
        als = ALSBaseline(p=1.5)
        with pytest.raises(ValueError, match="p must be in"):
            als.transform(X)

    def test_als_invalid_max_iter(self):
        """Test ALS with invalid max_iter."""
        X = np.random.randn(5, 50)
        als = ALSBaseline(max_iter=-1)
        with pytest.raises(ValueError, match="max_iter must be positive"):
            als.transform(X)

    def test_als_wrong_dimensions(self):
        """Test ALS with wrong input dimensions."""
        X = np.random.randn(100)
        als = ALSBaseline()
        with pytest.raises(ValueError, match="X must be 2D"):
            als.transform(X)

    def test_als_3d_input(self):
        """Test ALS rejects 3D input."""
        X = np.random.randn(5, 10, 20)
        als = ALSBaseline()
        with pytest.raises(ValueError, match="X must be 2D"):
            als.transform(X)


class TestRubberbandBaseline:
    """Tests for Rubberband baseline correction."""

    def test_rubberband_init(self):
        """Test Rubberband initialization."""
        rb = RubberbandBaseline()
        assert hasattr(rb, 'fit')
        assert hasattr(rb, 'transform')

    def test_rubberband_transform(self):
        """Test Rubberband transform on data."""
        X = np.random.randn(5, 100) + 3.0
        rb = RubberbandBaseline()
        Y = rb.transform(X)
        assert Y.shape == X.shape

    def test_rubberband_single_sample(self):
        """Test Rubberband on single sample."""
        X = np.ones((1, 50))
        rb = RubberbandBaseline()
        Y = rb.transform(X)
        assert Y.shape == X.shape


class TestPolynomialBaseline:
    """Tests for Polynomial baseline correction."""

    def test_polynomial_init_default(self):
        """Test Polynomial baseline init."""
        pb = PolynomialBaseline()
        assert hasattr(pb, 'fit')
        assert hasattr(pb, 'transform')

    def test_polynomial_transform(self):
        """Test Polynomial baseline transform."""
        X = np.random.randn(5, 100)
        pb = PolynomialBaseline()
        Y = pb.transform(X)
        assert Y.shape == X.shape


class TestVectorNormalizer:
    """Tests for Vector normalization."""

    def test_vector_norm_l2_default(self):
        """Test L2 normalization (default)."""
        X = np.array([[3.0, 4.0], [1.0, 1.0]])
        vn = VectorNormalizer(norm='l2')
        Y = vn.transform(X)
        
        # Check L2 norm of each row
        norms = np.linalg.norm(Y, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_vector_norm_l1(self):
        """Test L1 normalization."""
        X = np.array([[2.0, 3.0], [1.0, 1.0]])
        vn = VectorNormalizer(norm='l1')
        Y = vn.transform(X)
        
        # Check L1 norm of each row
        norms = np.sum(np.abs(Y), axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_vector_norm_max(self):
        """Test max normalization."""
        X = np.array([[2.0, 4.0], [1.0, 3.0]])
        vn = VectorNormalizer(norm='max')
        Y = vn.transform(X)
        
        # Check max abs of each row
        max_vals = np.max(np.abs(Y), axis=1)
        np.testing.assert_array_almost_equal(max_vals, [1.0, 1.0])

    def test_vector_norm_invalid_norm(self):
        """Test invalid norm type."""
        X = np.random.randn(5, 10)
        vn = VectorNormalizer(norm='invalid')
        with pytest.raises(ValueError, match="norm must be one of"):
            vn.transform(X)

    def test_vector_norm_wrong_dims(self):
        """Test wrong input dimensions."""
        X = np.random.randn(100)
        vn = VectorNormalizer()
        with pytest.raises(ValueError, match="X must be 2D"):
            vn.transform(X)

    def test_vector_norm_zero_vector(self):
        """Test with zero vector (should handle gracefully)."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        vn = VectorNormalizer()
        Y = vn.transform(X)
        assert np.all(np.isfinite(Y))


class TestAreaNormalizer:
    """Tests for Area normalization."""

    def test_area_normalizer_init(self):
        """Test AreaNormalizer initialization."""
        an = AreaNormalizer()
        assert hasattr(an, 'fit')
        assert hasattr(an, 'transform')

    def test_area_normalizer_transform(self):
        """Test AreaNormalizer transform."""
        X = np.random.randn(5, 100)
        an = AreaNormalizer()
        Y = an.transform(X)
        assert Y.shape == X.shape


class TestInternalPeakNormalizer:
    """Tests for Internal peak normalization."""

    def test_internal_peak_normalizer(self):
        """Test InternalPeakNormalizer."""
        X = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        
        # Requires target_wavenumber parameter
        ipn = InternalPeakNormalizer(target_wavenumber=2000.0)
        Y = ipn.fit(X, wavenumbers=wavenumbers).transform(X, wavenumbers)
        assert Y.shape == X.shape


class TestSNVNormalizer:
    """Tests for Standard Normal Variate normalization."""

    def test_snv_transform(self):
        """Test SNV transform."""
        X = np.random.randn(5, 100)
        snv = SNVNormalizer()
        Y = snv.transform(X)
        assert Y.shape == X.shape
        
        # Check that variance of each row is ~1
        variances = np.var(Y, axis=1)
        np.testing.assert_array_almost_equal(variances, np.ones(5), decimal=0)

    def test_snv_zero_variance(self):
        """Test SNV with zero variance spectrum."""
        X = np.ones((5, 100))
        snv = SNVNormalizer()
        Y = snv.transform(X)
        assert np.all(np.isfinite(Y))


class TestMSCNormalizer:
    """Tests for Multiplicative Scatter Correction."""

    def test_msc_fit_transform(self):
        """Test MSC fit and transform."""
        X_train = np.random.randn(5, 100)
        X_test = np.random.randn(3, 100)
        
        msc = MSCNormalizer()
        msc.fit(X_train)
        Y = msc.transform(X_test)
        assert Y.shape == X_test.shape

    def test_msc_transform_without_fit(self):
        """Test MSC transform without fit raises error."""
        X = np.random.randn(5, 100)
        msc = MSCNormalizer()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            msc.transform(X)


class TestSavitzkyGolaySmoother:
    """Tests for Savitzky-Golay smoothing."""

    def test_savitzky_golay_init(self):
        """Test SG init."""
        sg = SavitzkyGolaySmoother(window_length=7, polyorder=3)
        assert sg.window_length == 7
        assert sg.polyorder == 3

    def test_savitzky_golay_transform(self):
        """Test SG transform."""
        X = np.random.randn(5, 100)
        sg = SavitzkyGolaySmoother(window_length=7, polyorder=2)
        Y = sg.transform(X)
        assert Y.shape == X.shape

    def test_savitzky_golay_noise_reduction(self):
        """Test that SG reduces noise."""
        # Create noisy signal
        x = np.linspace(0, 4*np.pi, 100)
        signal = np.sin(x)
        noise = np.random.randn(100) * 0.1
        X = (signal + noise).reshape(1, -1)
        
        sg = SavitzkyGolaySmoother(window_length=7, polyorder=2)
        Y = sg.transform(X)
        
        # Smoothed should be closer to original signal
        mse_original = np.mean((X.ravel() - signal) ** 2)
        mse_smoothed = np.mean((Y.ravel() - signal) ** 2)
        assert mse_smoothed < mse_original

    def test_savitzky_golay_invalid_window(self):
        """Test SG with invalid window length."""
        X = np.random.randn(5, 100)
        sg = SavitzkyGolaySmoother(window_length=6)  # Even window
        with pytest.raises(ValueError, match="window_length must be"):
            sg.transform(X)

    def test_savitzky_golay_polyorder_too_high(self):
        """Test SG with polyorder >= window_length."""
        X = np.random.randn(5, 100)
        sg = SavitzkyGolaySmoother(window_length=5, polyorder=5)
        with pytest.raises(ValueError, match="polyorder must be less"):
            sg.transform(X)


class TestMovingAverageSmoother:
    """Tests for Moving average smoothing."""

    def test_moving_average_init(self):
        """Test MA init."""
        ma = MovingAverageSmoother(window_size=5)
        assert ma.window_size == 5

    def test_moving_average_transform(self):
        """Test MA transform."""
        X = np.random.randn(5, 100)
        ma = MovingAverageSmoother(window_size=5)
        Y = ma.transform(X)
        assert Y.shape == X.shape

    def test_moving_average_noise_reduction(self):
        """Test that MA reduces noise."""
        # Create noisy signal
        x = np.linspace(0, 4*np.pi, 100)
        signal = np.sin(x)
        noise = np.random.randn(100) * 0.1
        X = (signal + noise).reshape(1, -1)
        
        ma = MovingAverageSmoother(window_size=5)
        Y = ma.transform(X)
        
        # Should reduce variance
        assert np.var(Y) < np.var(X)



class TestSpikeCorrectionModule:
    """Tests for spike (cosmic ray) detection and correction."""

    def test_cosmic_ray_correct_basic(self):
        """Test cosmic ray correction."""
        X = np.random.randn(5, 100)
        # Add spikes
        X[0, 50] = 100.0
        X[2, 75] = -100.0
        
        X_corr, report = correct_cosmic_rays(X, window=5)
        assert X_corr.shape == X.shape
        assert report.total_spikes >= 0
        assert len(report.spikes_per_spectrum) == X.shape[0]

    def test_cosmic_ray_detect_spikes(self):
        """Test spike detection works."""
        X = np.ones((1, 100))
        X[0, 50] = 100.0  # Clear spike
        
        X_corr, report = correct_cosmic_rays(X, window=5)
        # Should detect at least one spike
        assert report.spikes_per_spectrum[0] > 0


class TestRangeCropper:
    """Tests for wavenumber/range cropping."""

    def test_cropper_init(self):
        """Test RangeCropper init."""
        cropper = RangeCropper(min_wn=400, max_wn=4000)
        assert cropper.min_wn == 400
        assert cropper.max_wn == 4000

    def test_cropper_invalid_range(self):
        """Test RangeCropper with invalid range."""
        with pytest.raises(ValueError, match="min_wn must be less"):
            RangeCropper(min_wn=4000, max_wn=400)

    def test_cropper_equal_bounds(self):
        """Test RangeCropper with equal bounds."""
        with pytest.raises(ValueError, match="min_wn must be less"):
            RangeCropper(min_wn=400, max_wn=400)

    def test_cropper_crop(self):
        """Test wavenumber cropping."""
        wn = np.linspace(400, 4000, 100)
        X = np.random.randn(5, 100)
        
        cropper = RangeCropper(min_wn=800, max_wn=3000)
        cropper.fit(X, wavenumbers=wn)
        X_crop, wn_crop = cropper.transform(X, wn)
        
        assert X_crop.shape[1] < X.shape[1]
        assert np.all(wn_crop >= 800)
        assert np.all(wn_crop <= 3000)

    def test_cropper_full_range(self):
        """Test cropping with full range."""
        wn = np.linspace(400, 4000, 100)
        X = np.random.randn(5, 100)
        
        cropper = RangeCropper(min_wn=400, max_wn=4000)
        X_crop, wn_crop = cropper.transform(X, wn)
        
        assert X_crop.shape == X.shape
        assert len(wn_crop) == len(wn)


class TestPreprocessingPipeline:
    """Integration tests for preprocessing pipelines."""

    def test_baseline_normalization_pipeline(self):
        """Test baseline correction followed by normalization."""
        X = np.random.randn(5, 100) + 5.0
        
        als = ALSBaseline(lambda_=1e4, p=0.01, max_iter=3)
        X_baseline = als.transform(X)
        
        vn = VectorNormalizer(norm='l2')
        X_final = vn.transform(X_baseline)
        
        assert X_final.shape == X.shape
        norms = np.linalg.norm(X_final, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0]*5)

    def test_smoothing_normalization_pipeline(self):
        """Test smoothing followed by normalization."""
        X = np.random.randn(5, 100)
        
        sg = SavitzkyGolaySmoother(window_length=7, polyorder=2)
        X_smooth = sg.transform(X)
        
        snv = SNVNormalizer()
        X_final = snv.transform(X_smooth)
        
        assert X_final.shape == X.shape

    def test_spike_removal_baseline_pipeline(self):
        """Test spike removal followed by baseline correction."""
        X = np.random.randn(5, 100)
        X[:, 50] += 100  # Add spikes
        
        X_despike, _ = correct_cosmic_rays(X, window=5)
        
        als = ALSBaseline(lambda_=1e4, p=0.01, max_iter=3)
        X_final = als.transform(X_despike)
        
        assert X_final.shape == X.shape

    def test_full_preprocessing_pipeline(self):
        """Test full preprocessing pipeline."""
        X = np.random.randn(5, 100) + 2.0
        X[:, 50] += 50  # Add spike
        
        # Pipeline: despike -> baseline -> smooth -> normalize
        X, _ = correct_cosmic_rays(X, window=5)
        
        als = ALSBaseline(lambda_=1e4, p=0.01, max_iter=3)
        X = als.transform(X)
        
        sg = SavitzkyGolaySmoother(window_length=7, polyorder=2)
        X = sg.transform(X)
        
        vn = VectorNormalizer(norm='l2')
        X = vn.transform(X)
        
        assert X.shape == (5, 100)
        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0]*5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
