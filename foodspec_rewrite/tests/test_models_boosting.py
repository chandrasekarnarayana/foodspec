"""
Comprehensive tests for gradient boosting classifier wrappers.

Tests cover both XGBoost and LightGBM with optional dependency handling:
- Clear error messages when dependencies are missing
- Functional tests when dependencies are installed
- All standard classifier functionality
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# Test helpers for dependency checking
def is_xgboost_available():
    """Check if XGBoost is installed."""
    try:
        import xgboost

        return True
    except ImportError:
        return False


def is_lightgbm_available():
    """Check if LightGBM is installed."""
    try:
        import lightgbm

        return True
    except ImportError:
        return False


XGBOOST_AVAILABLE = is_xgboost_available()
LIGHTGBM_AVAILABLE = is_lightgbm_available()


# ============================================================================
# XGBoost Tests
# ============================================================================


class TestXGBoostImportError:
    """Test XGBoost import/usage depending on availability."""

    def test_clear_error_message_on_import(self):
        """Test XGBoost handling - error when not installed, success when installed."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        if XGBOOST_AVAILABLE:
            # When installed, verify it can be instantiated
            model = XGBoostClassifierWrapper(n_estimators=10)
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict_proba')
        else:
            # When not installed, verify clear error message
            with pytest.raises(ImportError) as excinfo:
                XGBoostClassifierWrapper()

            error_msg = str(excinfo.value)
            assert "XGBoost is not installed" in error_msg
            assert "pip install xgboost" in error_msg
            assert "pip install foodspec[boosting]" in error_msg

    def test_error_message_actionable(self):
        """Test XGBoost provides actionable info."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        if XGBOOST_AVAILABLE:
            # When installed, verify basic fit/predict works
            import numpy as np
            model = XGBoostClassifierWrapper(n_estimators=10, random_state=42)
            X = np.random.randn(50, 10)
            y = np.random.randint(0, 2, 50)
            model.fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape == (50, 2)
        else:
            # When not installed, verify error message is actionable
            with pytest.raises(ImportError) as excinfo:
                XGBoostClassifierWrapper()

            error_msg = str(excinfo.value)
            assert "pip install" in error_msg


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostDefault:
    """Test XGBoostClassifierWrapper with default parameters."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_default_instantiation(self):
        """Test instantiation with default parameters."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        clf = XGBoostClassifierWrapper()
        assert clf.n_estimators == 100
        assert clf.max_depth == 6
        assert clf.learning_rate == 0.1
        assert clf.random_state == 0

    def test_fit_predict_basic(self, binary_data):
        """Test basic fit and predict."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper()
        clf.fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds) <= {0, 1}

    def test_deterministic_with_seed(self, binary_data):
        """Test reproducibility with same seed."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data

        clf1 = XGBoostClassifierWrapper(random_state=42)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)

        clf2 = XGBoostClassifierWrapper(random_state=42)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostPredictProba:
    """Test XGBoostClassifierWrapper predict_proba method."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_predict_proba_shape(self, binary_data):
        """Test predict_proba returns correct shape."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        """Test probabilities sum to 1 for each sample."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_predict_proba_bounds(self, binary_data):
        """Test probabilities are in [0, 1]."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_proba_not_fitted_raises(self):
        """Test predict_proba raises before fitting."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        clf = XGBoostClassifierWrapper()
        X = np.random.randn(10, 50)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(X)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostFeatureImportance:
    """Test XGBoostClassifierWrapper feature importance."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_get_feature_importance_shape(self, binary_data):
        """Test feature importance has correct shape."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        assert importance.shape == (50,)

    def test_feature_importance_sums_to_one(self, binary_data):
        """Test feature importances sum to 1."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        total = importance.sum()
        np.testing.assert_allclose(total, 1.0, rtol=1e-5)

    def test_feature_importance_non_negative(self, binary_data):
        """Test feature importances are non-negative."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        assert np.all(importance >= 0.0)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostParameterValidation:
    """Test XGBoostClassifierWrapper parameter validation."""

    def test_invalid_n_estimators(self):
        """Test negative n_estimators raises error."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        with pytest.raises(ValueError, match="n_estimators must be positive"):
            XGBoostClassifierWrapper(n_estimators=0)

    def test_invalid_max_depth(self):
        """Test invalid max_depth raises error."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        with pytest.raises(ValueError, match="max_depth must be positive"):
            XGBoostClassifierWrapper(max_depth=0)

    def test_invalid_learning_rate_low(self):
        """Test learning_rate <= 0 raises error."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        with pytest.raises(ValueError, match="learning_rate must be in"):
            XGBoostClassifierWrapper(learning_rate=0.0)

    def test_invalid_learning_rate_high(self):
        """Test learning_rate > 1 raises error."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        with pytest.raises(ValueError, match="learning_rate must be in"):
            XGBoostClassifierWrapper(learning_rate=1.5)

    def test_invalid_subsample(self):
        """Test invalid subsample raises error."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        with pytest.raises(ValueError, match="subsample must be in"):
            XGBoostClassifierWrapper(subsample=1.5)

    def test_invalid_colsample_bytree(self):
        """Test invalid colsample_bytree raises error."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        with pytest.raises(ValueError, match="colsample_bytree must be in"):
            XGBoostClassifierWrapper(colsample_bytree=0.0)

    def test_invalid_reg_alpha(self):
        """Test negative reg_alpha raises error."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        with pytest.raises(ValueError, match="reg_alpha must be non-negative"):
            XGBoostClassifierWrapper(reg_alpha=-1.0)

    def test_invalid_scale_pos_weight(self):
        """Test invalid scale_pos_weight raises error."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        with pytest.raises(ValueError, match="scale_pos_weight must be positive"):
            XGBoostClassifierWrapper(scale_pos_weight=-1.0)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostFactoryMethods:
    """Test XGBoostClassifierWrapper factory methods."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_fast_constructor(self, binary_data):
        """Test fast factory method."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper.fast()
        clf.fit(X, y)

        assert clf.n_estimators == 50
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_strong_regularization_constructor(self, binary_data):
        """Test strong_regularization factory method."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper.strong_regularization()
        clf.fit(X, y)

        assert clf.max_depth == 3
        assert clf.reg_alpha == 1.0
        assert clf.reg_lambda == 10.0
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_weak_regularization_constructor(self, binary_data):
        """Test weak_regularization factory method."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper.weak_regularization()
        clf.fit(X, y)

        assert clf.max_depth == 10
        preds = clf.predict(X)
        assert preds.shape == (100,)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostSaveLoad:
    """Test XGBoostClassifierWrapper save/load functionality."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_save_unfitted_raises(self):
        """Test save raises error for unfitted model."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        clf = XGBoostClassifierWrapper()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            with pytest.raises(RuntimeError, match="Cannot save unfitted"):
                clf.save(path)

    def test_save_load_roundtrip(self, binary_data):
        """Test save and load roundtrip."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        X, y = binary_data
        clf = XGBoostClassifierWrapper()
        clf.fit(X, y)

        preds1 = clf.predict(X)
        proba1 = clf.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf_loaded = XGBoostClassifierWrapper.load(path)
            preds2 = clf_loaded.predict(X)
            proba2 = clf_loaded.predict_proba(X)

        np.testing.assert_array_equal(preds1, preds2)
        np.testing.assert_allclose(proba1, proba2, rtol=1e-5)


# ============================================================================
# LightGBM Tests
# ============================================================================


class TestLightGBMImportError:
    """Test LightGBM import/usage depending on availability."""

    def test_clear_error_message_on_import(self):
        """Test LightGBM handling - error when not installed, success when installed."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        if LIGHTGBM_AVAILABLE:
            # When installed, verify it can be instantiated
            model = LightGBMClassifierWrapper(n_estimators=10)
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict_proba')
        else:
            # When not installed, verify clear error message
            with pytest.raises(ImportError) as excinfo:
                LightGBMClassifierWrapper()

            error_msg = str(excinfo.value)
            assert "LightGBM is not installed" in error_msg
            assert "pip install lightgbm" in error_msg
            assert "pip install foodspec[boosting]" in error_msg

    def test_error_message_actionable(self):
        """Test LightGBM provides actionable info."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        if LIGHTGBM_AVAILABLE:
            # When installed, verify basic fit/predict works
            import numpy as np
            model = LightGBMClassifierWrapper(n_estimators=10, random_state=42)
            X = np.random.randn(50, 10)
            y = np.random.randint(0, 2, 50)
            model.fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape == (50, 2)
        else:
            # When not installed, verify error message is actionable
            with pytest.raises(ImportError) as excinfo:
                LightGBMClassifierWrapper()

            error_msg = str(excinfo.value)
            assert "pip install" in error_msg


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMDefault:
    """Test LightGBMClassifierWrapper with default parameters."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_default_instantiation(self):
        """Test instantiation with default parameters."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        clf = LightGBMClassifierWrapper()
        assert clf.n_estimators == 100
        assert clf.max_depth == -1
        assert clf.learning_rate == 0.1
        assert clf.random_state == 0

    def test_fit_predict_basic(self, binary_data):
        """Test basic fit and predict."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper()
        clf.fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds) <= {0, 1}

    def test_deterministic_with_seed(self, binary_data):
        """Test reproducibility with same seed."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data

        clf1 = LightGBMClassifierWrapper(random_state=42)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)

        clf2 = LightGBMClassifierWrapper(random_state=42)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMPredictProba:
    """Test LightGBMClassifierWrapper predict_proba method."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_predict_proba_shape(self, binary_data):
        """Test predict_proba returns correct shape."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        """Test probabilities sum to 1 for each sample."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_predict_proba_bounds(self, binary_data):
        """Test probabilities are in [0, 1]."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_proba_not_fitted_raises(self):
        """Test predict_proba raises before fitting."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        clf = LightGBMClassifierWrapper()
        X = np.random.randn(10, 50)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(X)


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMFeatureImportance:
    """Test LightGBMClassifierWrapper feature importance."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_get_feature_importance_shape(self, binary_data):
        """Test feature importance has correct shape."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        assert importance.shape == (50,)

    def test_feature_importance_sums_to_one(self, binary_data):
        """Test feature importances sum to 1."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        total = importance.sum()
        np.testing.assert_allclose(total, 1.0, rtol=1e-5)

    def test_feature_importance_non_negative(self, binary_data):
        """Test feature importances are non-negative."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper()
        clf.fit(X, y)

        importance = clf.get_feature_importance()
        assert np.all(importance >= 0.0)


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMParameterValidation:
    """Test LightGBMClassifierWrapper parameter validation."""

    def test_invalid_n_estimators(self):
        """Test negative n_estimators raises error."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        with pytest.raises(ValueError, match="n_estimators must be positive"):
            LightGBMClassifierWrapper(n_estimators=0)

    def test_invalid_max_depth(self):
        """Test invalid max_depth raises error."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        with pytest.raises(ValueError, match="max_depth must be positive or -1"):
            LightGBMClassifierWrapper(max_depth=0)

    def test_invalid_learning_rate_low(self):
        """Test learning_rate <= 0 raises error."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        with pytest.raises(ValueError, match="learning_rate must be in"):
            LightGBMClassifierWrapper(learning_rate=0.0)

    def test_invalid_num_leaves(self):
        """Test invalid num_leaves raises error."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        with pytest.raises(ValueError, match="num_leaves must be > 1"):
            LightGBMClassifierWrapper(num_leaves=1)

    def test_invalid_subsample(self):
        """Test invalid subsample raises error."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        with pytest.raises(ValueError, match="subsample must be in"):
            LightGBMClassifierWrapper(subsample=1.5)

    def test_invalid_class_weight(self):
        """Test invalid class_weight raises error."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        with pytest.raises(ValueError, match="class_weight must be None or 'balanced'"):
            LightGBMClassifierWrapper(class_weight="invalid")


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMFactoryMethods:
    """Test LightGBMClassifierWrapper factory methods."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_fast_constructor(self, binary_data):
        """Test fast factory method."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper.fast()
        clf.fit(X, y)

        assert clf.n_estimators == 50
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_strong_regularization_constructor(self, binary_data):
        """Test strong_regularization factory method."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper.strong_regularization()
        clf.fit(X, y)

        assert clf.max_depth == 5
        assert clf.num_leaves == 15
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_weak_regularization_constructor(self, binary_data):
        """Test weak_regularization factory method."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper.weak_regularization()
        clf.fit(X, y)

        assert clf.max_depth == -1
        assert clf.num_leaves == 63
        preds = clf.predict(X)
        assert preds.shape == (100,)


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMSaveLoad:
    """Test LightGBMClassifierWrapper save/load functionality."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_save_unfitted_raises(self):
        """Test save raises error for unfitted model."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        clf = LightGBMClassifierWrapper()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            with pytest.raises(RuntimeError, match="Cannot save unfitted"):
                clf.save(path)

    def test_save_load_roundtrip(self, binary_data):
        """Test save and load roundtrip."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        X, y = binary_data
        clf = LightGBMClassifierWrapper()
        clf.fit(X, y)

        preds1 = clf.predict(X)
        proba1 = clf.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf_loaded = LightGBMClassifierWrapper.load(path)
            preds2 = clf_loaded.predict(X)
            proba2 = clf_loaded.predict_proba(X)

        np.testing.assert_array_equal(preds1, preds2)
        np.testing.assert_allclose(proba1, proba2, rtol=1e-5)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostIntegration:
    """Integration tests for XGBoostClassifierWrapper."""

    def test_imbalanced_workflow(self):
        """Test XGBoost on imbalanced data."""
        from foodspec.models.boosting import XGBoostClassifierWrapper

        np.random.seed(42)
        X = np.random.randn(100, 500)  # High-dimensional
        y = np.array([0] * 80 + [1] * 20)  # Imbalanced

        clf = XGBoostClassifierWrapper(scale_pos_weight=4.0)
        clf.fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        importance = clf.get_feature_importance()

        assert preds.shape == (100,)
        assert proba.shape == (100, 2)
        assert importance.shape == (500,)


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMIntegration:
    """Integration tests for LightGBMClassifierWrapper."""

    def test_imbalanced_workflow(self):
        """Test LightGBM on imbalanced data."""
        from foodspec.models.boosting import LightGBMClassifierWrapper

        np.random.seed(42)
        X = np.random.randn(100, 500)  # High-dimensional
        y = np.array([0] * 80 + [1] * 20)  # Imbalanced

        clf = LightGBMClassifierWrapper(class_weight="balanced")
        clf.fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        importance = clf.get_feature_importance()

        assert preds.shape == (100,)
        assert proba.shape == (100, 2)
        assert importance.shape == (500,)
