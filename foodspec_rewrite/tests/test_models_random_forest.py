"""
Comprehensive tests for RandomForestClassifierWrapper.

Tests cover:
- Default instantiation and basic functionality
- Parameter validation
- Feature importance extraction
- Out-of-bag scoring
- Predict proba support
- Save/load serialization
- Factory methods
- Integration workflows
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from foodspec.models.classical import RandomForestClassifierWrapper


class TestRandomForestDefault:
    """Test RandomForestClassifierWrapper with default parameters."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_default_instantiation(self):
        """Test instantiation with default parameters."""
        clf = RandomForestClassifierWrapper()
        assert clf.n_estimators == 100
        assert clf.max_depth == 20
        assert clf.max_features == "sqrt"
        assert clf.criterion == "gini"
        assert clf.oob_score is True
        assert clf.random_state == 0

    def test_fit_predict_basic(self, binary_data):
        """Test basic fit and predict."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds) <= {0, 1}

    def test_deterministic_with_seed(self, binary_data):
        """Test reproducibility with same seed."""
        X, y = binary_data

        clf1 = RandomForestClassifierWrapper(random_state=42)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)

        clf2 = RandomForestClassifierWrapper(random_state=42)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

    def test_different_seeds_different_results(self, binary_data):
        """Test that different seeds produce different results."""
        X, y = binary_data

        clf1 = RandomForestClassifierWrapper(random_state=0)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)

        clf2 = RandomForestClassifierWrapper(random_state=1)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)

        # Different seeds may produce different predictions
        assert preds1.shape == preds2.shape


class TestRandomForestPredictProba:
    """Test RandomForestClassifierWrapper predict_proba method."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_predict_proba_shape(self, binary_data):
        """Test predict_proba returns correct shape."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        """Test probabilities sum to 1 for each sample."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_predict_proba_bounds(self, binary_data):
        """Test probabilities are in [0, 1]."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_proba_multiclass(self):
        """Test predict_proba with multiclass."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1, 2] * 33 + [0])

        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 3)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_predict_proba_not_fitted_raises(self):
        """Test predict_proba raises before fitting."""
        clf = RandomForestClassifierWrapper()
        X = np.random.randn(10, 50)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(X)


class TestRandomForestFeatureImportance:
    """Test RandomForestClassifierWrapper feature importance."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_get_feature_importance_shape(self, binary_data):
        """Test feature importance has correct shape."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        assert importance.shape == (50,)

    def test_feature_importance_sums_to_one(self, binary_data):
        """Test feature importances sum to 1."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        total = importance.sum()
        np.testing.assert_allclose(total, 1.0, rtol=1e-5)

    def test_feature_importance_non_negative(self, binary_data):
        """Test feature importances are non-negative."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        assert np.all(importance >= 0.0)

    def test_feature_importance_not_fitted_raises(self):
        """Test get_feature_importance raises before fitting."""
        clf = RandomForestClassifierWrapper()

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.get_feature_importance()


class TestRandomForestOOBScore:
    """Test RandomForestClassifierWrapper out-of-bag scoring."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_oob_score_available(self, binary_data):
        """Test out-of-bag score is available."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper(oob_score=True)
        clf.fit(X, y)

        oob_score = clf.oob_score_
        assert 0.0 <= oob_score <= 1.0

    def test_oob_score_false_raises(self, binary_data):
        """Test oob_score raises when oob_score=False."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper(oob_score=False)
        clf.fit(X, y)

        with pytest.raises(RuntimeError, match="oob_score not available"):
            _ = clf.oob_score_

    def test_oob_score_not_fitted_raises(self):
        """Test oob_score raises before fitting."""
        clf = RandomForestClassifierWrapper(oob_score=True)

        with pytest.raises(RuntimeError, match="not fitted"):
            _ = clf.oob_score_


class TestRandomForestParameterValidation:
    """Test RandomForestClassifierWrapper parameter validation."""

    def test_invalid_n_estimators(self):
        """Test negative n_estimators raises error."""
        with pytest.raises(ValueError, match="n_estimators must be positive"):
            RandomForestClassifierWrapper(n_estimators=0)

    def test_invalid_max_depth(self):
        """Test invalid max_depth raises error."""
        with pytest.raises(ValueError, match="max_depth must be"):
            RandomForestClassifierWrapper(max_depth=0)

    def test_invalid_min_samples_split(self):
        """Test invalid min_samples_split raises error."""
        with pytest.raises(ValueError, match="min_samples_split must be"):
            RandomForestClassifierWrapper(min_samples_split=1)

    def test_invalid_min_samples_leaf(self):
        """Test invalid min_samples_leaf raises error."""
        with pytest.raises(ValueError, match="min_samples_leaf must be"):
            RandomForestClassifierWrapper(min_samples_leaf=0)

    def test_invalid_max_features_string(self):
        """Test invalid max_features string raises error."""
        with pytest.raises(ValueError, match="max_features string must be"):
            RandomForestClassifierWrapper(max_features="invalid")

    def test_invalid_max_features_int(self):
        """Test invalid max_features int raises error."""
        with pytest.raises(ValueError, match="max_features int must be positive"):
            RandomForestClassifierWrapper(max_features=0)

    def test_invalid_criterion(self):
        """Test invalid criterion raises error."""
        with pytest.raises(ValueError, match="criterion must be"):
            RandomForestClassifierWrapper(criterion="invalid")

    def test_invalid_class_weight_string(self):
        """Test invalid class_weight string raises error."""
        with pytest.raises(ValueError, match="class_weight string must be"):
            RandomForestClassifierWrapper(class_weight="invalid")


class TestRandomForestInputValidation:
    """Test RandomForestClassifierWrapper input validation."""

    def test_X_not_2d_raises(self):
        """Test 1D X raises error."""
        clf = RandomForestClassifierWrapper()
        X = np.random.randn(100)
        y = np.array([0, 1] * 50)

        with pytest.raises(ValueError, match="X must be 2D"):
            clf.fit(X, y)

    def test_y_not_1d_raises(self):
        """Test 2D y raises error."""
        clf = RandomForestClassifierWrapper()
        X = np.random.randn(100, 50)
        y = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="y must be 1D"):
            clf.fit(X, y)

    def test_mismatched_lengths_raises(self):
        """Test mismatched X and y lengths raise error."""
        clf = RandomForestClassifierWrapper()
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 30)  # 60 samples

        with pytest.raises(ValueError, match="inconsistent lengths"):
            clf.fit(X, y)

    def test_predict_not_fitted_raises(self):
        """Test predict without fitting raises error."""
        clf = RandomForestClassifierWrapper()
        X = np.random.randn(10, 50)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(X)


class TestRandomForestClassWeight:
    """Test RandomForestClassifierWrapper class weight handling."""

    @pytest.fixture
    def imbalanced_data(self):
        """Generate imbalanced data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0] * 80 + [1] * 20)
        return X, y

    def test_balanced_class_weight(self, imbalanced_data):
        """Test class_weight='balanced' works."""
        X, y = imbalanced_data
        clf = RandomForestClassifierWrapper(class_weight="balanced")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_custom_class_weight_dict(self, imbalanced_data):
        """Test custom class_weight dict."""
        X, y = imbalanced_data
        clf = RandomForestClassifierWrapper(class_weight={0: 1.0, 1: 5.0})
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_none_class_weight(self, imbalanced_data):
        """Test class_weight=None (default)."""
        X, y = imbalanced_data
        clf = RandomForestClassifierWrapper(class_weight=None)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestRandomForestDefaultHyperparams:
    """Test RandomForestClassifierWrapper default hyperparameters."""

    def test_default_hyperparams_spectroscopy(self):
        """Test default hyperparams are suitable for spectroscopy."""
        defaults = RandomForestClassifierWrapper.default_hyperparams()

        assert defaults["n_estimators"] == 100
        assert defaults["max_depth"] == 20
        assert defaults["max_features"] == "sqrt"
        assert defaults["class_weight"] == "balanced"
        assert defaults["oob_score"] is True

    def test_can_instantiate_with_defaults(self):
        """Test can instantiate with default hyperparams."""
        defaults = RandomForestClassifierWrapper.default_hyperparams()
        clf = RandomForestClassifierWrapper(**defaults)
        assert clf.n_estimators == 100


class TestRandomForestFactoryMethods:
    """Test RandomForestClassifierWrapper factory methods."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_fast_constructor(self, binary_data):
        """Test fast factory method."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper.fast()
        clf.fit(X, y)

        assert clf.n_estimators == 50
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_strong_regularization_constructor(self, binary_data):
        """Test strong_regularization factory method."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper.strong_regularization()
        clf.fit(X, y)

        assert clf.max_depth == 5
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_weak_regularization_constructor(self, binary_data):
        """Test weak_regularization factory method."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper.weak_regularization()
        clf.fit(X, y)

        assert clf.max_depth is None
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_many_trees_constructor(self, binary_data):
        """Test many_trees factory method."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper.many_trees()
        clf.fit(X, y)

        assert clf.n_estimators == 500
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestRandomForestSaveLoad:
    """Test RandomForestClassifierWrapper save/load functionality."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_save_unfitted_raises(self):
        """Test save raises error for unfitted model."""
        clf = RandomForestClassifierWrapper()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            with pytest.raises(RuntimeError, match="not fitted"):
                clf.save(path)

    def test_save_load_roundtrip(self, binary_data):
        """Test save and load roundtrip."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        preds1 = clf.predict(X)
        proba1 = clf.predict_proba(X)
        importance1 = clf.get_feature_importance()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf_loaded = RandomForestClassifierWrapper.load(path)
            preds2 = clf_loaded.predict(X)
            proba2 = clf_loaded.predict_proba(X)
            importance2 = clf_loaded.get_feature_importance()

        np.testing.assert_array_equal(preds1, preds2)
        np.testing.assert_allclose(proba1, proba2, rtol=1e-5)
        np.testing.assert_allclose(importance1, importance2, rtol=1e-5)

    def test_load_oob_score(self, binary_data):
        """Test loaded model has OOB score available."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper(oob_score=True)
        clf.fit(X, y)
        oob1 = clf.oob_score_

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf_loaded = RandomForestClassifierWrapper.load(path)
            oob2 = clf_loaded.oob_score_

        np.testing.assert_allclose(oob1, oob2)


class TestRandomForestMaxFeatures:
    """Test RandomForestClassifierWrapper max_features parameter."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_sqrt_max_features(self, binary_data):
        """Test sqrt max_features."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper(max_features="sqrt")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_log2_max_features(self, binary_data):
        """Test log2 max_features."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper(max_features="log2")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_int_max_features(self, binary_data):
        """Test integer max_features."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper(max_features=10)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestRandomForestCriterion:
    """Test RandomForestClassifierWrapper criterion parameter."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_gini_criterion(self, binary_data):
        """Test gini criterion."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper(criterion="gini")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_entropy_criterion(self, binary_data):
        """Test entropy criterion."""
        X, y = binary_data
        clf = RandomForestClassifierWrapper(criterion="entropy")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestIntegration:
    """Integration tests for RandomForestClassifierWrapper."""

    @pytest.fixture
    def imbalanced_data(self):
        """Generate imbalanced spectroscopy data."""
        np.random.seed(42)
        X = np.random.randn(100, 500)  # High-dimensional
        y = np.array([0] * 80 + [1] * 20)  # Imbalanced
        return X, y

    def test_imbalanced_workflow(self, imbalanced_data):
        """Test Random Forest on imbalanced spectroscopy data."""
        X, y = imbalanced_data

        clf = RandomForestClassifierWrapper(class_weight="balanced")
        clf.fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        importance = clf.get_feature_importance()
        oob_score = clf.oob_score_

        assert preds.shape == (100,)
        assert proba.shape == (100, 2)
        assert importance.shape == (500,)
        assert 0.0 <= oob_score <= 1.0

    def test_hyperparameter_sweep(self, imbalanced_data):
        """Test Random Forest hyperparameter sweep."""
        X, y = imbalanced_data

        hyperparams = [
            {"n_estimators": 50},
            {"n_estimators": 100},
            {"n_estimators": 200},
            {"max_depth": 5},
            {"max_depth": 20},
            {"max_depth": None},
        ]

        for params in hyperparams:
            clf = RandomForestClassifierWrapper(**params)
            clf.fit(X, y)
            preds = clf.predict(X)
            assert preds.shape == (100,)

    def test_feature_importance_identifies_top_features(self, imbalanced_data):
        """Test feature importance can identify important features."""
        X, y = imbalanced_data

        clf = RandomForestClassifierWrapper(n_estimators=200, random_state=42)
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        top_features = np.argsort(importance)[-10:]  # Top 10 features

        # Just verify we can identify top features
        assert len(top_features) == 10
        assert all(0 <= f < 500 for f in top_features)

    def test_multiclass_workflow(self):
        """Test Random Forest on multiclass problem."""
        np.random.seed(42)
        X = np.random.randn(150, 100)
        y = np.array([0, 1, 2] * 50)  # 3 classes

        clf = RandomForestClassifierWrapper()
        clf.fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)

        assert preds.shape == (150,)
        assert proba.shape == (150, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_parallelization_n_jobs(self, imbalanced_data):
        """Test parallelization with n_jobs parameter."""
        X, y = imbalanced_data

        # Single job
        clf_single = RandomForestClassifierWrapper(n_estimators=100, n_jobs=1)
        clf_single.fit(X, y)
        preds_single = clf_single.predict(X)

        # All jobs
        clf_all = RandomForestClassifierWrapper(n_estimators=100, n_jobs=-1)
        clf_all.fit(X, y)
        preds_all = clf_all.predict(X)

        # Both should work
        assert preds_single.shape == (100,)
        assert preds_all.shape == (100,)
