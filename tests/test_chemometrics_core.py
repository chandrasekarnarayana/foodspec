"""Tests for chemometrics core modules: alignment, unmixing, PLS, NNLS, stability, agreement, drift."""
import numpy as np
import pytest
from sklearn.datasets import make_regression, make_blobs
from sklearn.linear_model import Ridge

# Feature engineering
from foodspec.features.alignment import CrossCorrelationAligner, DynamicTimeWarpingAligner, align_spectra
from foodspec.features.unmixing import NNLSUnmixer, unmix_spectrum

# Modeling
from foodspec.modeling.chemometrics.pls import PLSRegression, VIPCalculator
from foodspec.modeling.chemometrics.nnls import NNLSRegression, ConstrainedLasso

# Validation
from foodspec.validation.stability import BootstrapStability, StabilityIndex
from foodspec.validation.agreement import BlandAltmanAnalysis, DemingRegression

# QC
from foodspec.qc.drift_ewma import EWMAControlChart, DriftDetector


class TestCrossCorrelationAligner:
    """Test cross-correlation alignment."""

    def test_fit_transform(self):
        """Test fit/transform API."""
        X = np.random.randn(10, 50)
        aligner = CrossCorrelationAligner(max_shift=5, reference_idx=0)
        X_aligned = aligner.fit_transform(X)
        
        assert X_aligned.shape == X.shape
        assert hasattr(aligner, 'reference_')
        assert hasattr(aligner, 'shifts_')

    def test_pipeline_compatible(self):
        """Test sklearn pipeline compatibility."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        X = np.random.randn(20, 30)
        pipeline = Pipeline([
            ('align', CrossCorrelationAligner(max_shift=3)),
            ('scale', StandardScaler()),
        ])
        
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape


class TestDynamicTimeWarpingAligner:
    """Test DTW alignment."""

    def test_fit_transform(self):
        """Test fit/transform API."""
        X = np.random.randn(10, 50)
        aligner = DynamicTimeWarpingAligner(window=10, reference_idx=0)
        X_aligned = aligner.fit_transform(X)
        
        assert X_aligned.shape == X.shape
        assert hasattr(aligner, 'warping_paths_')

    def test_different_window_size(self):
        """Test different window sizes."""
        X = np.random.randn(15, 100)
        for window in [5, 10, 20]:
            aligner = DynamicTimeWarpingAligner(window=window)
            X_aligned = aligner.fit_transform(X)
            assert X_aligned.shape == X.shape


class TestNNLSUnmixer:
    """Test NNLS unmixing."""

    def test_unmix_single_spectrum(self):
        """Test unmixing single spectrum."""
        # Library: 3 components x 10 wavenumbers
        library = np.random.rand(3, 10)
        # Single mixture: 10 wavenumbers
        mixture = np.random.rand(10)
        
        unmixer = NNLSUnmixer()
        unmixer.fit(library)
        coeff = unmixer.transform(mixture.reshape(1, -1))
        
        assert coeff.shape == (1, 3)
        assert np.all(coeff >= 0)  # Non-negative constraint

    def test_unmix_multiple_spectra(self):
        """Test unmixing multiple spectra."""
        library = np.random.rand(5, 10)  # 5 components, 10 wavenumbers
        mixtures = np.random.rand(20, 10)  # 20 mixtures
        
        unmixer = NNLSUnmixer()
        unmixer.fit(library)
        coeffs = unmixer.transform(mixtures)
        
        assert coeffs.shape == (20, 5)
        assert np.all(coeffs >= 0)

    def test_residual_computation(self):
        """Test residual computation."""
        library = np.array([[1.0, 2.0], [3.0, 1.0]])
        mixture = np.array([2.0, 2.0])
        
        unmixer = NNLSUnmixer()
        unmixer.fit(library.T)
        coeffs, residuals = unmixer.transform(mixture.reshape(1, -1), return_residual=True)
        
        assert residuals.shape == (1,)
        assert residuals[0] >= 0


class TestPLSRegression:
    """Test PLSR."""

    def test_fit_predict(self):
        """Test basic fit/predict."""
        X, y = make_regression(n_samples=50, n_features=20, noise=0.1)
        
        pls = PLSRegression(n_components=3, scale=True)
        pls.fit(X, y)
        y_pred = pls.predict(X)
        
        assert y_pred.shape == y.shape
        assert hasattr(pls, 'coef_')

    def test_component_selection(self):
        """Test different number of components."""
        X, y = make_regression(n_samples=100, n_features=30)
        
        for n_comp in [1, 5, 10]:
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X, y)
            assert pls.x_scores_.shape[1] == n_comp

    def test_vip_calculation(self):
        """Test VIP score calculation."""
        X, y = make_regression(n_samples=50, n_features=20)
        
        vip = VIPCalculator.calculate_vip(X, y, n_components=3)
        
        assert vip.shape == (20,)
        assert np.all(vip > 0)


class TestNNLSRegression:
    """Test NNLS regression."""

    def test_fit_predict(self):
        """Test basic fit/predict."""
        X = np.abs(np.random.randn(50, 10))
        y = np.abs(X @ np.random.rand(10))
        
        nnls_reg = NNLSRegression(scale=True)
        nnls_reg.fit(X, y)
        y_pred = nnls_reg.predict(X)
        
        assert y_pred.shape == y.shape
        assert np.all(nnls_reg.coef_ >= 0)

    def test_non_negativity_constraint(self):
        """Test that coefficients are non-negative."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        nnls_reg = NNLSRegression()
        nnls_reg.fit(X, y)
        
        assert np.all(nnls_reg.coef_ >= -1e-10)  # Accounting for numerical precision


class TestConstrainedLasso:
    """Test Constrained LASSO."""

    def test_fit_predict(self):
        """Test fit/predict."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        
        classo = ConstrainedLasso(alpha=0.01)
        classo.fit(X, y)
        y_pred = classo.predict(X)
        
        assert y_pred.shape == y.shape
        assert np.all(classo.coef_ >= 0)

    def test_sparsity(self):
        """Test sparsity calculation."""
        X = np.random.randn(50, 30)
        y = np.random.randn(50)
        
        classo = ConstrainedLasso(alpha=0.1)
        classo.fit(X, y)
        sparsity = classo.sparsity()
        
        assert 0 <= sparsity <= 1


class TestBootstrapStability:
    """Test bootstrap stability analysis."""

    def test_parameter_stability(self):
        """Test parameter stability assessment."""
        X, y = make_regression(n_samples=50, n_features=10)
        
        bs = BootstrapStability(n_bootstrap=20, random_state=42)
        param_mean, param_std, param_ci = bs.assess_parameter_stability(
            X, y,
            fit_func=lambda x, yy: Ridge().fit(x, yy),
            param_func=lambda m: m.coef_
        )
        
        assert param_mean.shape == (10,)
        assert param_std.shape == (10,)
        assert param_ci.shape == (10, 2)
        assert np.all(param_std >= 0)

    def test_prediction_stability(self):
        """Test prediction stability."""
        X, y = make_regression(n_samples=50, n_features=10)
        X_test = np.random.randn(5, 10)
        
        bs = BootstrapStability(n_bootstrap=20, random_state=42)
        pred_mean, pred_std, pred_ci = bs.assess_prediction_stability(
            X, y,
            fit_func=lambda x, yy: Ridge().fit(x, yy),
            X_test=X_test
        )
        
        assert pred_mean.shape == (5,)
        assert pred_std.shape == (5,)
        assert pred_ci.shape == (5, 2)


class TestStabilityIndex:
    """Test stability indices."""

    def test_jackknife_resampling(self):
        """Test jackknife resampling."""
        X, y = make_regression(n_samples=30, n_features=10)
        
        param_mean, param_std = StabilityIndex.jackknife_resampling(
            X, y,
            fit_func=lambda x, yy: Ridge().fit(x, yy),
            param_func=lambda m: m.coef_
        )
        
        assert param_mean.shape == (10,)
        assert param_std.shape == (10,)

    def test_stability_ratio(self):
        """Test parameter stability ratio."""
        params = np.array([1.0, 2.0, -0.5, 3.0])
        stds = np.array([0.1, 0.2, 0.1, 0.3])
        
        ratio = StabilityIndex.parameter_stability_ratio(stds, params)
        
        assert ratio.shape == (4,)
        assert np.all(ratio > 0)

    def test_reproducibility_index(self):
        """Test reproducibility index."""
        y1 = np.array([1.0, 2.0, 3.0, 4.0])
        y2 = np.array([1.1, 2.05, 2.95, 3.9])
        
        r_rep = StabilityIndex.model_reproducibility_index(y1, y2)
        
        assert 0 <= r_rep <= 1


class TestBlandAltmanAnalysis:
    """Test Bland-Altman agreement analysis."""

    def test_calculate(self):
        """Test Bland-Altman calculation."""
        method1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        method2 = np.array([1.1, 2.05, 2.95, 3.9, 5.1])
        
        ba = BlandAltmanAnalysis(confidence=0.95)
        mean_diff, std_diff, ll, ul, corr = ba.calculate(method1, method2)
        
        assert isinstance(mean_diff, float)
        assert std_diff >= 0
        assert ll < ul
        assert -1 <= corr <= 1

    def test_report_generation(self):
        """Test report generation."""
        m1 = np.random.randn(20) + 5
        m2 = m1 + np.random.randn(20) * 0.1
        
        ba = BlandAltmanAnalysis()
        ba.calculate(m1, m2)
        report = ba.get_report()
        
        assert "Bland-Altman" in report
        assert "Mean Difference" in report


class TestDemingRegression:
    """Test Deming regression."""

    def test_fit_predict(self):
        """Test fit/predict."""
        X = np.linspace(1, 10, 20)
        y = 2.0 * X + 1.0 + np.random.randn(20) * 0.5
        
        deming = DemingRegression(variance_ratio=1.0)
        deming.fit(X, y)
        y_pred = deming.predict(X)
        
        assert y_pred.shape == (20,)

    def test_concordance_correlation(self):
        """Test concordance correlation."""
        X = np.random.randn(30)
        y = X + np.random.randn(30) * 0.1
        
        deming = DemingRegression()
        deming.fit(X, y)
        ccc = deming.get_concordance_correlation(X, y)
        
        assert -1 <= ccc <= 1


class TestEWMAControlChart:
    """Test EWMA control chart."""

    def test_initialize(self):
        """Test initialization."""
        X_ref = np.random.randn(30, 1)
        
        ewma = EWMAControlChart(lambda_=0.2, confidence=0.99)
        ewma.initialize(X_ref)
        
        assert hasattr(ewma, 'center_')
        assert hasattr(ewma, 'ucl_')
        assert hasattr(ewma, 'lcl_')

    def test_update(self):
        """Test update."""
        X_ref = np.random.randn(30, 2)
        ewma = EWMAControlChart()
        ewma.initialize(X_ref)
        
        x_new = np.random.randn(2)
        ewma_val, is_alarm = ewma.update(x_new)
        
        assert ewma_val.shape == (2,)
        assert isinstance(is_alarm, (bool, np.bool_))

    def test_process_stream(self):
        """Test processing stream."""
        X_ref = np.random.randn(20, 1)
        ewma = EWMAControlChart()
        ewma.initialize(X_ref)
        
        X_stream = np.random.randn(10, 1)
        ewma_vals, alarms = ewma.process(X_stream)
        
        assert ewma_vals.shape == (10, 1)
        assert alarms.shape == (10,)


class TestDriftDetector:
    """Test drift detection."""

    def test_initialize(self):
        """Test initialization."""
        X_ref = np.random.randn(50, 3)
        
        dd = DriftDetector(lambda_=0.2, min_samples=20)
        dd.initialize(X_ref)
        
        assert hasattr(dd, 'ewma_chart_')
        assert hasattr(dd, 'reference_mean_')

    def test_check_drift(self):
        """Test drift check."""
        X_ref = np.random.randn(50, 3)
        dd = DriftDetector()
        dd.initialize(X_ref)
        
        x_new = np.random.randn(3)
        result = dd.check_drift(x_new)
        
        assert 'ewma_alarm' in result
        assert 'mahalanobis_distance' in result
        assert 'drift_type' in result
        assert result['drift_type'] in ['none', 'shift', 'outlier']

    def test_process_stream(self):
        """Test processing stream."""
        X_ref = np.random.randn(40, 2)
        dd = DriftDetector()
        dd.initialize(X_ref)
        
        X_stream = np.random.randn(15, 2)
        results = dd.process_stream(X_stream)
        
        assert len(results) == 15

    def test_drift_summary(self):
        """Test drift summary."""
        X_ref = np.random.randn(30, 2)
        dd = DriftDetector()
        dd.initialize(X_ref)
        
        X_stream = np.random.randn(10, 2)
        dd.process_stream(X_stream)
        summary = dd.get_drift_summary()
        
        assert 'n_observations' in summary
        assert 'alarm_rate' in summary
        assert 0 <= summary['alarm_rate'] <= 1


class TestIntegration:
    """Integration tests across modules."""

    def test_alignment_pipeline(self):
        """Test alignment in feature extraction pipeline."""
        X = np.random.randn(20, 100)
        X_aligned = align_spectra(X, method="xcorr", max_shift=5)
        assert X_aligned.shape == X.shape

    def test_unmixing_pipeline(self):
        """Test unmixing in feature extraction."""
        library = np.random.rand(5, 50)
        mixtures = np.random.rand(10, 50)
        
        coeffs = unmix_spectrum(mixtures, library)
        assert coeffs.shape == (10, 5)
        assert np.all(coeffs >= 0)

    def test_pls_vip_pipeline(self):
        """Test PLSR with VIP calculation."""
        X, y = make_regression(n_samples=100, n_features=30)
        
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)
        y_pred = pls.predict(X)
        
        vip = VIPCalculator.calculate_vip(X, y, n_components=5)
        
        assert y_pred.shape == y.shape
        assert vip.shape == (30,)

    def test_agreement_analysis_pipeline(self):
        """Test agreement analysis workflow."""
        method1 = np.random.randn(30) + 5
        method2 = method1 + np.random.randn(30) * 0.2
        
        # Bland-Altman
        ba = BlandAltmanAnalysis()
        ba.calculate(method1, method2)
        
        # Deming
        deming = DemingRegression()
        deming.fit(method1, method2)
        
        assert deming.slope_ > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
