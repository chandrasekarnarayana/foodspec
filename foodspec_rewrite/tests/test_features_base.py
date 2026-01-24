"""
Tests for feature engineering base classes and FeatureSet container.

Verifies:
- FeatureExtractor ABC interface conformance
- FeatureSet validation and shape constraints
- get_params/set_params sklearn compatibility
- feature_names length matches n_features
"""

import numpy as np
import pytest

from foodspec.features.base import FeatureExtractor, FeatureSet


class DummyFeatureExtractor(FeatureExtractor):
    """Minimal concrete implementation for testing."""

    def __init__(self, n_components: int = 3, scale: bool = True):
        self.n_components = n_components
        self.scale = scale
        self._fitted = False

    def fit(self, X, y=None, x_grid=None, meta=None):
        """Dummy fit."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        self._fitted = True
        self._n_features_in = X.shape[1]
        return self

    def transform(self, X, x_grid=None, meta=None):
        """Dummy transform: return first n_components columns."""
        if not self._fitted:
            raise RuntimeError("Not fitted")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self._n_features_in:
            raise ValueError("Feature shape mismatch")
        
        Xf = X[:, : self.n_components]
        feature_names = [f"feat_{i}" for i in range(self.n_components)]
        return Xf, feature_names


class TestFeatureExtractorInterface:
    """Test FeatureExtractor ABC."""

    def test_cannot_instantiate_abstract_base(self):
        """Verify FeatureExtractor is abstract."""
        with pytest.raises(TypeError):
            FeatureExtractor()

    def test_concrete_implementation(self):
        """Test concrete extractor implements all required methods."""
        extractor = DummyFeatureExtractor(n_components=5)
        
        # Check methods exist
        assert hasattr(extractor, "fit")
        assert hasattr(extractor, "transform")
        assert hasattr(extractor, "fit_transform")
        assert hasattr(extractor, "get_params")
        assert hasattr(extractor, "set_params")

    def test_fit_transform_workflow(self):
        """Test standard fit/transform workflow."""
        X_train = np.random.randn(50, 20)
        X_test = np.random.randn(30, 20)
        
        extractor = DummyFeatureExtractor(n_components=5)
        
        # Fit on train
        extractor.fit(X_train)
        assert extractor._fitted
        
        # Transform train and test
        Xf_train, names_train = extractor.transform(X_train)
        Xf_test, names_test = extractor.transform(X_test)
        
        assert Xf_train.shape == (50, 5)
        assert Xf_test.shape == (30, 5)
        assert len(names_train) == 5
        assert len(names_test) == 5
        assert names_train == names_test

    def test_fit_transform_convenience(self):
        """Test fit_transform convenience method."""
        X = np.random.randn(40, 15)
        
        extractor = DummyFeatureExtractor(n_components=3)
        Xf, names = extractor.fit_transform(X)
        
        assert Xf.shape == (40, 3)
        assert len(names) == 3
        assert names == ["feat_0", "feat_1", "feat_2"]

    def test_transform_before_fit_error(self):
        """Test error when transform called before fit."""
        X = np.random.randn(20, 10)
        extractor = DummyFeatureExtractor(n_components=2)
        
        with pytest.raises(RuntimeError, match="Not fitted"):
            extractor.transform(X)

    def test_get_params(self):
        """Test get_params returns public attributes."""
        extractor = DummyFeatureExtractor(n_components=7, scale=False)
        params = extractor.get_params()
        
        assert params["n_components"] == 7
        assert params["scale"] is False
        # Private attributes should not be in params
        assert "_fitted" not in params

    def test_set_params(self):
        """Test set_params modifies attributes."""
        extractor = DummyFeatureExtractor(n_components=3, scale=True)
        
        # Modify params
        extractor.set_params(n_components=10, scale=False)
        
        assert extractor.n_components == 10
        assert extractor.scale is False

    def test_set_params_returns_self(self):
        """Test set_params returns self for chaining."""
        extractor = DummyFeatureExtractor()
        result = extractor.set_params(n_components=5)
        
        assert result is extractor

    def test_optional_parameters(self):
        """Test optional x_grid and meta parameters."""
        X = np.random.randn(30, 50)
        x_grid = np.linspace(1000, 2000, 50)
        meta = {"sample_ids": ["s1", "s2", "s3"]}
        
        extractor = DummyFeatureExtractor(n_components=4)
        
        # Fit and transform with optional params
        extractor.fit(X, y=None, x_grid=x_grid, meta=meta)
        Xf, names = extractor.transform(X, x_grid=x_grid, meta=meta)
        
        assert Xf.shape == (30, 4)
        assert len(names) == 4


class TestFeatureSet:
    """Test FeatureSet container."""

    def test_creation_basic(self):
        """Test basic FeatureSet creation."""
        Xf = np.random.randn(10, 3)
        names = ["f1", "f2", "f3"]
        
        fs = FeatureSet(Xf=Xf, feature_names=names)
        
        assert fs.n_samples == 10
        assert fs.n_features == 3
        assert fs.feature_names == ["f1", "f2", "f3"]
        assert isinstance(fs.feature_meta, dict)
        assert len(fs.feature_meta) == 0

    def test_creation_with_metadata(self):
        """Test FeatureSet with metadata."""
        Xf = np.random.randn(20, 5)
        names = ["pc1", "pc2", "pc3", "pc4", "pc5"]
        meta = {
            "explained_variance_ratio": [0.45, 0.23, 0.12, 0.08, 0.05],
            "extractor_type": "pca",
        }
        
        fs = FeatureSet(Xf=Xf, feature_names=names, feature_meta=meta)
        
        assert fs.n_samples == 20
        assert fs.n_features == 5
        assert fs.feature_meta["extractor_type"] == "pca"
        assert len(fs.feature_meta["explained_variance_ratio"]) == 5

    def test_shape_constraint_n_features(self):
        """Test Xf.shape[1] must equal len(feature_names)."""
        Xf = np.random.randn(10, 3)
        
        # Correct length
        fs = FeatureSet(Xf=Xf, feature_names=["a", "b", "c"])
        assert fs.n_features == 3
        
        # Incorrect length
        with pytest.raises(ValueError, match="must match n_features"):
            FeatureSet(Xf=Xf, feature_names=["a", "b"])
        
        with pytest.raises(ValueError, match="must match n_features"):
            FeatureSet(Xf=Xf, feature_names=["a", "b", "c", "d"])

    def test_xf_must_be_2d(self):
        """Test Xf must be 2D array."""
        # 1D array should fail
        with pytest.raises(ValueError, match="must be 2D"):
            FeatureSet(Xf=np.array([1, 2, 3]), feature_names=["a", "b", "c"])
        
        # 3D array should fail
        with pytest.raises(ValueError, match="must be 2D"):
            FeatureSet(Xf=np.random.randn(10, 5, 2), feature_names=["a"])

    def test_xf_must_be_numpy(self):
        """Test Xf must be numpy array."""
        with pytest.raises(TypeError, match="must be numpy array"):
            FeatureSet(Xf=[[1, 2], [3, 4]], feature_names=["a", "b"])

    def test_feature_names_must_be_list(self):
        """Test feature_names must be a list."""
        Xf = np.random.randn(5, 2)
        
        with pytest.raises(TypeError, match="must be list"):
            FeatureSet(Xf=Xf, feature_names=("a", "b"))

    def test_select_features(self):
        """Test feature selection by indices."""
        Xf = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        names = ["a", "b", "c", "d", "e"]
        meta = {"info": "test"}
        
        fs = FeatureSet(Xf=Xf, feature_names=names, feature_meta=meta)
        
        # Select indices [0, 2, 4]
        subset = fs.select_features([0, 2, 4])
        
        assert subset.n_samples == 2
        assert subset.n_features == 3
        assert subset.feature_names == ["a", "c", "e"]
        np.testing.assert_array_equal(subset.Xf, [[1, 3, 5], [6, 8, 10]])
        assert subset.feature_meta["info"] == "test"

    def test_concatenate(self):
        """Test horizontal concatenation of FeatureSets."""
        Xf1 = np.array([[1, 2], [3, 4], [5, 6]])
        names1 = ["a", "b"]
        meta1 = {"extractor1": "pca"}
        fs1 = FeatureSet(Xf=Xf1, feature_names=names1, feature_meta=meta1)
        
        Xf2 = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        names2 = ["c", "d", "e"]
        meta2 = {"extractor2": "peaks"}
        fs2 = FeatureSet(Xf=Xf2, feature_names=names2, feature_meta=meta2)
        
        # Concatenate
        combined = fs1.concatenate(fs2)
        
        assert combined.n_samples == 3
        assert combined.n_features == 5
        assert combined.feature_names == ["a", "b", "c", "d", "e"]
        np.testing.assert_array_equal(
            combined.Xf,
            [[1, 2, 10, 20, 30], [3, 4, 40, 50, 60], [5, 6, 70, 80, 90]],
        )
        assert combined.feature_meta["extractor1"] == "pca"
        assert combined.feature_meta["extractor2"] == "peaks"

    def test_concatenate_n_samples_mismatch(self):
        """Test concatenate fails with n_samples mismatch."""
        fs1 = FeatureSet(Xf=np.random.randn(10, 2), feature_names=["a", "b"])
        fs2 = FeatureSet(Xf=np.random.randn(5, 3), feature_names=["c", "d", "e"])
        
        with pytest.raises(ValueError, match="n_samples mismatch"):
            fs1.concatenate(fs2)


class TestAcceptanceCriteria:
    """Verify acceptance tests from requirements."""

    def test_acceptance_shape_constraint(self):
        """✅ FeatureSet.Xf.shape == (n_samples, n_features)"""
        Xf = np.random.randn(25, 7)
        names = [f"f{i}" for i in range(7)]
        
        fs = FeatureSet(Xf=Xf, feature_names=names)
        
        assert fs.Xf.shape == (25, 7)
        assert fs.n_samples == 25
        assert fs.n_features == 7

    def test_acceptance_feature_names_length(self):
        """✅ len(feature_names) == n_features"""
        Xf = np.random.randn(15, 4)
        names = ["a", "b", "c", "d"]
        
        fs = FeatureSet(Xf=Xf, feature_names=names)
        
        assert len(fs.feature_names) == fs.n_features
        assert len(fs.feature_names) == 4

    def test_acceptance_enforced_at_creation(self):
        """✅ Length constraint enforced at creation time."""
        Xf = np.random.randn(10, 5)
        
        # Must fail if lengths don't match
        with pytest.raises(ValueError):
            FeatureSet(Xf=Xf, feature_names=["a", "b", "c"])  # 3 != 5


__all__ = [
    "TestFeatureExtractorInterface",
    "TestFeatureSet",
    "TestAcceptanceCriteria",
]
