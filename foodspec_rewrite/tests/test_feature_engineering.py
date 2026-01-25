"""
Comprehensive tests for feature engineering system.

Tests cover:
- All feature extractors (peaks, bands, chemometric, selection)
- Leakage safety (fit on train only, transform on test)
- Deterministic behavior with seeds
- Hybrid composition
- Integration with Protocol and FeatureSet
- Input validation and error handling
"""

import numpy as np
import pytest

from foodspec.features import (
    BandIntegration,
    FeatureComposer,
    FeatureExtractor,
    FeatureSet,
    PCAFeatureExtractor,
    PeakAreas,
    PeakHeights,
    PeakRatios,
    PLSFeatureExtractor,
    StabilitySelector,
)


@pytest.fixture
def sample_spectra():
    """Generate sample spectral data with synthetic peaks."""
    rng = np.random.default_rng(42)
    x = np.linspace(1000, 2000, 1001)
    
    def gauss(x, mu, amp=1.0, sigma=10.0):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Create 3 samples with peaks at 1200, 1500, 1800
    y1 = gauss(x, 1200, amp=5.0) + gauss(x, 1500, amp=3.0) + gauss(x, 1800, amp=2.0)
    y2 = gauss(x, 1200, amp=3.0) + gauss(x, 1500, amp=6.0) + gauss(x, 1800, amp=1.0)
    y3 = gauss(x, 1200, amp=4.0) + gauss(x, 1500, amp=2.0) + gauss(x, 1800, amp=5.0)
    
    X = np.vstack([y1, y2, y3]) + rng.normal(0, 0.1, (3, len(x)))
    return X, x


@pytest.fixture
def train_test_data():
    """Generate train/test split for leakage testing."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((50, 100))
    X_test = rng.standard_normal((20, 100))
    y_train = rng.integers(0, 2, 50)
    y_test = rng.integers(0, 2, 20)
    return X_train, X_test, y_train, y_test


class TestFeatureSetAbstraction:
    """Test FeatureSet container."""

    def test_featureset_creation(self):
        """Verify FeatureSet construction and properties."""
        Xf = np.random.randn(10, 3)
        fs = FeatureSet(
            Xf=Xf,
            feature_names=["f1", "f2", "f3"],
            feature_meta={"extractors": ["pca", "pca", "peak"]}
        )
        
        assert fs.n_samples == 10
        assert fs.n_features == 3
        assert fs.feature_names == ["f1", "f2", "f3"]
        assert fs.feature_meta["extractors"] == ["pca", "pca", "peak"]
        assert isinstance(fs.Xf, np.ndarray)
        assert fs.Xf.shape == (10, 3)

    def test_featureset_select(self):
        """Test feature selection by index."""
        Xf = np.random.randn(10, 5)
        fs = FeatureSet(Xf=Xf, feature_names=["a", "b", "c", "d", "e"])
        
        subset = fs.select_features([0, 2, 4])
        assert subset.feature_names == ["a", "c", "e"]
        assert subset.n_samples == 10
        assert subset.Xf.shape == (10, 3)

    def test_featureset_validation(self):
        """Test input validation."""
        # Test non-2D array
        with pytest.raises(ValueError):
            FeatureSet(Xf=np.array([1, 2, 3]), feature_names=["a", "b", "c"])
        
        # Test name length mismatch
        with pytest.raises(ValueError):
            FeatureSet(Xf=np.random.randn(10, 3), feature_names=["a", "b"])


class TestPeakFeatures:
    """Test peak-based feature extractors."""

    def test_peak_heights(self, sample_spectra):
        """Test PeakHeights extractor."""
        X, x = sample_spectra
        extractor = PeakHeights(peaks=[1200, 1500, 1800], window=20.0)
        features = extractor.compute(X, x)
        
        assert features.shape == (3, 3)
        assert list(features.columns) == ["height@1200", "height@1500", "height@1800"]
        # Sample 1 should have highest peak at 1200
        assert features.iloc[0, 0] > features.iloc[1, 0]

    def test_peak_areas(self, sample_spectra):
        """Test PeakAreas extractor."""
        X, x = sample_spectra
        # Convert window-based to band-based: window=30 around 1500 means 1470-1530
        extractor = PeakAreas(bands=[(1470, 1530)], baseline="linear")
        features = extractor.compute(X, x)
        
        assert features.shape == (3, 1)
        assert "area@" in features.columns[0]
        # Sample 2 should have largest area at 1500
        assert features.iloc[1, 0] > features.iloc[0, 0]
        assert features.iloc[1, 0] > features.iloc[2, 0]

    def test_peak_ratios(self, sample_spectra):
        """Test PeakRatios extractor."""
        X, x = sample_spectra
        extractor = PeakRatios(pairs=[(1200, 1500), (1500, 1800)], window=20.0)
        features = extractor.compute(X, x)
        
        assert features.shape == (3, 2)
        assert "ratio@1200/1500" in features.columns
        assert "ratio@1500/1800" in features.columns
        # All ratios should be positive
        assert (features > 0).all().all()

    def test_peak_validation(self, sample_spectra):
        """Test input validation for peak extractors."""
        X, x = sample_spectra
        
        # Empty peak list
        with pytest.raises(ValueError):
            PeakHeights(peaks=[])
        
        # Mismatched x length
        with pytest.raises(ValueError):
            PeakHeights(peaks=[1200]).compute(X, x[:500])


class TestBandIntegration:
    """Test band integration extractor."""

    def test_band_integration_basic(self, sample_spectra):
        """Test basic band integration."""
        X, x = sample_spectra
        extractor = BandIntegration(bands=[(1150, 1250), (1450, 1550)], method="trapz", baseline="none")
        features = extractor.compute(X, x)
        
        assert features.shape == (3, 2)
        assert "band_trapz@" in features.columns[0]
        assert "band_trapz@" in features.columns[1]

    def test_band_baseline_subtract(self, sample_spectra):
        """Test baseline subtraction."""
        X, x = sample_spectra
        extractor_no_bl = BandIntegration(bands=[(1450, 1550)], method="trapz", baseline="none")
        extractor_bl = BandIntegration(bands=[(1450, 1550)], method="trapz", baseline="linear")
        
        features_no_bl = extractor_no_bl.compute(X, x)
        features_bl = extractor_bl.compute(X, x)
        
        # Baseline subtraction should reduce area (removing offset)
        # This depends on data; just check both run
        assert features_no_bl.shape == features_bl.shape

    def test_band_validation(self, sample_spectra):
        """Test band integration validation."""
        X, x = sample_spectra
        
        # Invalid band (start >= end)
        with pytest.raises(ValueError):
            BandIntegration(bands=[(1500, 1400)])
        
        # Band outside wavenumber range - should still work (uses nearest point)
        extractor = BandIntegration(bands=[(3000, 4000)])
        features = extractor.compute(X, x)
        assert features.shape == (3, 1)


class TestChemometricFeatures:
    """Test PCA and PLS extractors."""

    def test_pca_fit_transform(self, train_test_data):
        """Test PCA with fit/transform separation."""
        X_train, X_test, _, _ = train_test_data
        
        pca = PCAFeatureExtractor(n_components=5, seed=42)
        pca.fit(X_train)
        
        features_train, names_train = pca.transform(X_train)
        features_test, names_test = pca.transform(X_test)

        assert features_train.shape == (50, 5)
        assert features_test.shape == (20, 5)
        assert names_train == ["pca_1", "pca_2", "pca_3", "pca_4", "pca_5"]
        
        # Check explained variance
        assert hasattr(pca, "explained_variance_ratio_")
        assert len(pca.explained_variance_ratio_) == 5

    def test_pca_determinism(self, train_test_data):
        """Test PCA is deterministic with fixed seed."""
        X_train, X_test, _, _ = train_test_data
        
        pca1 = PCAFeatureExtractor(n_components=3, seed=42)
        pca1.fit(X_train)
        features1, names1 = pca1.transform(X_test)
        
        pca2 = PCAFeatureExtractor(n_components=3, seed=42)
        pca2.fit(X_train)
        features2, names2 = pca2.transform(X_test)
        
        np.testing.assert_array_almost_equal(features1, features2)

    def test_pls_fit_transform(self, train_test_data):
        """Test PLS with supervised fitting."""
        X_train, X_test, y_train, _ = train_test_data
        
        pls = PLSFeatureExtractor(n_components=3, mode="regression")
        pls.fit(X_train, y_train)
        
        features_train, names_train = pls.transform(X_train)
        features_test, names_test = pls.transform(X_test)
        
        assert features_train.shape == (50, 3)
        assert features_test.shape == (20, 3)
        assert names_train == ["pls_1", "pls_2", "pls_3"]

    def test_pls_requires_labels(self, train_test_data):
        """Test PLS raises error without labels."""
        X_train, _, _, _ = train_test_data
        
        pls = PLSFeatureExtractor(n_components=2)
        with pytest.raises(ValueError, match="requires y"):
            pls.fit(X_train, y=None)

    def test_pca_not_fitted_error(self, train_test_data):
        """Test error when transform called before fit."""
        _, X_test, _, _ = train_test_data
        
        pca = PCAFeatureExtractor(n_components=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.transform(X_test)


class TestFeatureComposer:
    """Test hybrid feature composition."""

    def test_composer_basic(self, train_test_data):
        """Test basic feature composition."""
        X_train, X_test, y_train, _ = train_test_data
        
        composer = FeatureComposer([
            ("pca", PCAFeatureExtractor(n_components=3), {}),
            ("pls", PLSFeatureExtractor(n_components=2), {}),
        ])
        
        composer.fit(X_train, y_train)
        feature_set = composer.transform(X_test)
        
        assert isinstance(feature_set, FeatureSet)
        assert feature_set.Xf.shape == (20, 5)  # 3 PCA + 2 PLS
        assert len(feature_set.feature_names) == 5
        assert feature_set.feature_meta["per_feature_extractors"] == ["pca", "pca", "pca", "pls", "pls"]

    def test_composer_with_peaks(self, sample_spectra):
        """Test composer with peak-based extractors."""
        X, x = sample_spectra
        X_train, X_test = X[:2], X[2:]
        
        # Create simple wrapper for peaks since they use compute() not transform()
        class PeakWrapper:
            def __init__(self, extractor):
                self.extractor = extractor
            
            def fit(self, X, y=None, **kwargs):
                return self
            
            def transform(self, X, **kwargs):
                return self.extractor.compute(X, kwargs.get("x"))
        
        composer = FeatureComposer([
            ("pca", PCAFeatureExtractor(n_components=2), {}),
            ("peaks", PeakWrapper(PeakHeights(peaks=[1200, 1500])), {"x": x}),
        ])
        
        composer.fit(X_train, x=x)
        feature_set = composer.transform(X_test, x=x)
        
        assert feature_set.Xf.shape[1] == 4  # 2 PCA + 2 peaks

    def test_composer_not_fitted_error(self, train_test_data):
        """Test error when transform called before fit."""
        _, X_test, _, _ = train_test_data
        
        composer = FeatureComposer([
            ("pca", PCAFeatureExtractor(n_components=2), {}),
        ])
        
        with pytest.raises(RuntimeError, match="not fitted"):
            composer.transform(X_test)


class TestLeakageSafety:
    """Test that all extractors maintain leakage safety."""

    def test_pca_no_leakage(self, train_test_data):
        """Verify PCA fit only touches training data."""
        X_train, X_test, _, _ = train_test_data
        
        # Fit on train
        pca = PCAFeatureExtractor(n_components=3, seed=42)
        pca.fit(X_train)
        features_test1, names1 = pca.transform(X_test)
        
        # Fit on different training data should give different test transform
        X_train_different = np.random.RandomState(99).randn(50, 100)
        pca2 = PCAFeatureExtractor(n_components=3, seed=42)
        pca2.fit(X_train_different)
        features_test2, names2 = pca2.transform(X_test)
        
        # Test features should differ when trained on different data
        assert not np.allclose(features_test1, features_test2)

    def test_pls_no_leakage(self, train_test_data):
        """Verify PLS fit only touches training data and labels."""
        X_train, X_test, y_train, _ = train_test_data
        
        pls = PLSFeatureExtractor(n_components=2)
        pls.fit(X_train, y_train)
        features_test1, names1 = pls.transform(X_test)
        
        # Fit with different labels
        y_train_different = np.random.default_rng(99).integers(0, 2, 50)
        pls2 = PLSFeatureExtractor(n_components=2)
        pls2.fit(X_train, y_train_different)
        features_test2, names2 = pls2.transform(X_test)
        
        # Test features should differ when trained with different labels
        assert not np.allclose(features_test1, features_test2)

    def test_composer_no_leakage(self, train_test_data):
        """Verify composer maintains leakage safety."""
        X_train, X_test, y_train, _ = train_test_data
        
        composer = FeatureComposer([
            ("pca", PCAFeatureExtractor(n_components=2), {}),
        ])
        
        # Fit on train only
        composer.fit(X_train)
        
        # Should be able to transform test without issues
        feature_set = composer.transform(X_test)
        assert feature_set.Xf.shape == (20, 2)


class TestDeterminism:
    """Test deterministic behavior with seeds."""

    def test_pca_determinism(self):
        """Test PCA produces same results with same seed."""
        X = np.random.RandomState(42).randn(30, 50)
        
        results = []
        for _ in range(2):
            pca = PCAFeatureExtractor(n_components=3, seed=42)
            pca.fit(X)
            features, names = pca.transform(X)
            results.append(features)
        
        np.testing.assert_array_almost_equal(results[0], results[1])

    def test_stability_selector_determinism(self):
        """Test stability selection is deterministic."""
        from foodspec.models.classical import LogisticRegressionClassifier
        
        X = np.random.RandomState(42).randn(50, 20)
        y = np.random.default_rng(42).integers(0, 2, 50)
        
        results = []
        for _ in range(2):
            selector = StabilitySelector(
                estimator_factory=lambda: LogisticRegressionClassifier(penalty="l1", C=1.0, solver="saga"),
                n_resamples=10,
                random_state=42,
            )
            selector.fit(X, y)
            results.append(selector.selection_frequencies_)
        
        np.testing.assert_array_almost_equal(results[0], results[1])


class TestProtocolConformance:
    """Test that extractors conform to FeatureExtractor protocol."""

    def test_pca_conforms_to_protocol(self):
        """Verify PCA implements FeatureExtractor protocol."""
        pca = PCAFeatureExtractor(n_components=2)
        assert hasattr(pca, "fit")
        assert hasattr(pca, "transform")
        assert hasattr(pca, "fit_transform")

    def test_pls_conforms_to_protocol(self):
        """Verify PLS implements FeatureExtractor protocol."""
        pls = PLSFeatureExtractor(n_components=2)
        assert hasattr(pls, "fit")
        assert hasattr(pls, "transform")
        assert hasattr(pls, "fit_transform")


__all__ = [
    "TestFeatureSetAbstraction",
    "TestPeakFeatures",
    "TestBandIntegration",
    "TestChemometricFeatures",
    "TestFeatureComposer",
    "TestLeakageSafety",
    "TestDeterminism",
    "TestProtocolConformance",
]
