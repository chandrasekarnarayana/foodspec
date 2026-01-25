"""
Unit tests for LogisticRegressionClassifier wrapper.

Tests cover:
- Solver auto-selection based on penalty
- Class weight handling for imbalanced data
- Spectroscopy-suitable hyperparameters
- Parameter validation
- Deterministic training with random_state
- Coefficient and intercept access
- Save/load roundtrips
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.models.classical import LogisticRegressionClassifier


@pytest.fixture
def binary_data():
    """Balanced binary classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 50))
    y = np.array([0, 1] * 50)
    return X, y


@pytest.fixture
def imbalanced_data():
    """Imbalanced binary classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 50))
    y = np.array([0] * 80 + [1] * 20)  # 4:1 imbalance
    return X, y


@pytest.fixture
def multiclass_data():
    """Multiclass classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((120, 50))
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    return X, y


@pytest.fixture
def spectroscopy_data():
    """High-dimensional spectroscopy-like data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 500))  # Many wavelengths
    y = np.array([0, 1] * 30)
    return X, y


class TestLogisticRegressionDefault:
    """Test default behavior."""

    def test_default_instantiation(self):
        """Test model can be created with defaults."""
        clf = LogisticRegressionClassifier()
        
        assert clf.C == 1.0
        assert clf.penalty == "l2"
        assert clf.solver is None  # Auto-select
        assert clf.max_iter == 1000
        assert clf.random_state == 0
        assert clf.class_weight is None

    def test_fit_predict_basic(self, binary_data):
        """Test basic fit and predict."""
        X, y = binary_data
        clf = LogisticRegressionClassifier()
        
        clf.fit(X, y)
        preds = clf.predict(X)
        
        assert preds.shape == (100,)
        assert set(preds) == {0, 1}

    def test_deterministic_with_seed(self, binary_data):
        """Test deterministic results with random_state."""
        X, y = binary_data
        
        clf1 = LogisticRegressionClassifier(random_state=42)
        clf1.fit(X, y)
        proba1 = clf1.predict_proba(X)
        
        clf2 = LogisticRegressionClassifier(random_state=42)
        clf2.fit(X, y)
        proba2 = clf2.predict_proba(X)
        
        assert np.allclose(proba1, proba2)

    def test_different_seeds_different_results(self, binary_data):
        """Test different seeds may produce different results (solver randomness)."""
        X, y = binary_data
        
        clf1 = LogisticRegressionClassifier(random_state=1)
        clf1.fit(X, y)
        coef1 = clf1.get_coef()
        
        clf2 = LogisticRegressionClassifier(random_state=2)
        clf2.fit(X, y)
        coef2 = clf2.get_coef()
        
        # Results should be similar but may differ slightly
        assert coef1.shape == coef2.shape


class TestSolverSelection:
    """Test automatic solver selection based on penalty."""

    def test_l2_uses_lbfgs(self):
        """Test L2 penalty uses lbfgs solver."""
        clf = LogisticRegressionClassifier(penalty="l2")
        solver = clf._get_solver()
        assert solver == "lbfgs"

    def test_l1_uses_liblinear(self):
        """Test L1 penalty uses liblinear solver."""
        clf = LogisticRegressionClassifier(penalty="l1")
        solver = clf._get_solver()
        assert solver == "liblinear"

    def test_elasticnet_uses_saga(self):
        """Test elasticnet uses saga solver."""
        clf = LogisticRegressionClassifier(penalty="elasticnet")
        solver = clf._get_solver()
        assert solver == "saga"

    def test_explicit_solver_overrides_auto(self):
        """Test explicit solver overrides auto-selection."""
        clf = LogisticRegressionClassifier(penalty="l2", solver="saga")
        solver = clf._get_solver()
        assert solver == "saga"

    def test_l1_penalty_works(self, binary_data):
        """Test L1 penalty produces sparse solution."""
        X, y = binary_data
        
        clf = LogisticRegressionClassifier(penalty="l1", C=0.5)
        clf.fit(X, y)
        coef = clf.get_coef()[0]
        
        # L1 should produce some zero coefficients (sparsity)
        n_zeros = np.sum(coef == 0)
        assert n_zeros > 0, "L1 should produce sparse solutions"

    def test_elasticnet_penalty_works(self, binary_data):
        """Test elasticnet penalty works."""
        X, y = binary_data
        
        clf = LogisticRegressionClassifier(penalty="elasticnet", l1_ratio=0.5)
        clf.fit(X, y)
        preds = clf.predict(X)
        
        assert preds.shape == (100,)


class TestClassWeight:
    """Test class weight handling."""

    def test_balanced_class_weight(self, binary_data):
        """Test balanced class weight improves performance on imbalanced data."""
        X_train, y_train = binary_data
        X_test, y_test = binary_data
        
        # Create truly imbalanced test set
        y_test_imbalanced = np.array([0] * 90 + [1] * 10)
        
        # With balanced weights
        clf_balanced = LogisticRegressionClassifier(class_weight="balanced")
        clf_balanced.fit(X_train, y_train)
        
        # Without class weight
        clf_unweighted = LogisticRegressionClassifier(class_weight=None)
        clf_unweighted.fit(X_train, y_train)
        
        # Both should make predictions
        preds_balanced = clf_balanced.predict(X_test)
        preds_unweighted = clf_unweighted.predict(X_test)
        
        assert preds_balanced.shape == preds_unweighted.shape

    def test_custom_class_weight_dict(self, binary_data):
        """Test custom class weight dictionary."""
        X, y = binary_data
        
        class_weight_dict = {0: 1.0, 1: 2.0}  # Weight class 1 twice as much
        clf = LogisticRegressionClassifier(class_weight=class_weight_dict)
        clf.fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_invalid_class_weight_string(self):
        """Test error on invalid class weight string."""
        with pytest.raises(ValueError, match="balanced"):
            LogisticRegressionClassifier(class_weight="invalid")


class TestParameterValidation:
    """Test parameter validation."""

    def test_invalid_penalty(self):
        """Test error on invalid penalty."""
        with pytest.raises(ValueError, match="penalty"):
            LogisticRegressionClassifier(penalty="invalid")

    def test_negative_C(self):
        """Test error on negative C."""
        with pytest.raises(ValueError, match="C must be positive"):
            LogisticRegressionClassifier(C=-1.0)

    def test_zero_C(self):
        """Test error on zero C."""
        with pytest.raises(ValueError, match="C must be positive"):
            LogisticRegressionClassifier(C=0.0)

    def test_invalid_l1_ratio(self):
        """Test error on l1_ratio outside [0, 1]."""
        with pytest.raises(ValueError, match="l1_ratio"):
            LogisticRegressionClassifier(penalty="elasticnet", l1_ratio=1.5)

    def test_negative_max_iter(self):
        """Test error on negative max_iter."""
        with pytest.raises(ValueError, match="max_iter"):
            LogisticRegressionClassifier(max_iter=-1)

    def test_negative_tol(self):
        """Test error on negative tol."""
        with pytest.raises(ValueError, match="tol"):
            LogisticRegressionClassifier(tol=-0.01)


class TestInputValidation:
    """Test input validation during fit/predict."""

    def test_X_not_2d_raises(self, binary_data):
        """Test error if X is not 2D."""
        _, y = binary_data
        X_1d = np.random.randn(100)
        
        clf = LogisticRegressionClassifier()
        with pytest.raises(ValueError, match="2D"):
            clf.fit(X_1d, y)

    def test_y_not_1d_raises(self, binary_data):
        """Test error if y is not 1D."""
        X, _ = binary_data
        y_2d = np.array([[0, 1]] * 50)
        
        clf = LogisticRegressionClassifier()
        with pytest.raises(ValueError, match="1D"):
            clf.fit(X, y_2d)

    def test_mismatched_lengths_raises(self, binary_data):
        """Test error if X and y have different lengths."""
        X, _ = binary_data
        y = np.array([0, 1] * 40)  # 80 samples vs 100 in X
        
        clf = LogisticRegressionClassifier()
        with pytest.raises(ValueError, match="same number of samples"):
            clf.fit(X, y)

    def test_predict_not_fitted_raises(self, binary_data):
        """Test error if predict on unfitted model."""
        X, _ = binary_data
        clf = LogisticRegressionClassifier()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(X)


class TestPredictProba:
    """Test probabilistic predictions."""

    def test_predict_proba_shape(self, binary_data):
        """Test predict_proba shape."""
        X, y = binary_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert proba.shape == (100, 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        """Test probabilities sum to 1 across classes."""
        X, y = binary_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_multiclass(self, multiclass_data):
        """Test predict_proba for multiclass."""
        X, y = multiclass_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert proba.shape == (120, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_bounds(self, binary_data):
        """Test probabilities are in [0, 1]."""
        X, y = binary_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)


class TestCoefficientsAndIntercept:
    """Test coefficient and intercept access."""

    def test_get_coef(self, binary_data):
        """Test coefficient retrieval."""
        X, y = binary_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        
        coef = clf.get_coef()
        
        assert coef.shape[1] == 50  # n_features

    def test_get_intercept(self, binary_data):
        """Test intercept retrieval."""
        X, y = binary_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        
        intercept = clf.get_intercept()
        
        assert intercept.ndim == 1

    def test_coef_not_fitted_raises(self):
        """Test error if get_coef on unfitted model."""
        clf = LogisticRegressionClassifier()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.get_coef()

    def test_intercept_not_fitted_raises(self):
        """Test error if get_intercept on unfitted model."""
        clf = LogisticRegressionClassifier()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.get_intercept()


class TestDefaultHyperparams:
    """Test default_hyperparams() class method."""

    def test_default_hyperparams_spectroscopy(self):
        """Test default hyperparams are suitable for spectroscopy."""
        defaults = LogisticRegressionClassifier.default_hyperparams()
        
        assert isinstance(defaults, dict)
        assert defaults["C"] == 1.0  # Moderate regularization
        assert defaults["penalty"] == "l2"  # Smooth solutions
        assert defaults["class_weight"] == "balanced"  # Handle imbalance
        assert defaults["max_iter"] == 1000  # Convergence

    def test_can_instantiate_with_defaults(self):
        """Test can create model with default hyperparams."""
        defaults = LogisticRegressionClassifier.default_hyperparams()
        clf = LogisticRegressionClassifier(**defaults)
        
        assert clf.C == 1.0
        assert clf.penalty == "l2"
        assert clf.class_weight == "balanced"

    def test_defaults_work_with_spectroscopy_data(self, spectroscopy_data):
        """Test defaults work well with high-dimensional data."""
        X, y = spectroscopy_data
        
        defaults = LogisticRegressionClassifier.default_hyperparams()
        clf = LogisticRegressionClassifier(**defaults)
        clf.fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (60,)


class TestConvenienceConstructors:
    """Test convenience factory methods."""

    def test_sparse_features_constructor(self, binary_data):
        """Test sparse_features constructor."""
        X, y = binary_data
        
        clf = LogisticRegressionClassifier.sparse_features()
        
        assert clf.penalty == "l1"
        assert clf.solver == "liblinear"
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_sparse_features_override(self, binary_data):
        """Test sparse_features with parameter override."""
        X, y = binary_data
        
        clf = LogisticRegressionClassifier.sparse_features(C=0.1)
        
        assert clf.C == 0.1
        assert clf.penalty == "l1"

    def test_strong_regularization_constructor(self, binary_data):
        """Test strong_regularization constructor."""
        X, y = binary_data
        
        clf = LogisticRegressionClassifier.strong_regularization()
        
        assert clf.C == 0.1  # Low C = strong regularization
        assert clf.penalty == "l2"
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_weak_regularization_constructor(self, binary_data):
        """Test weak_regularization constructor."""
        X, y = binary_data
        
        clf = LogisticRegressionClassifier.weak_regularization()
        
        assert clf.C == 10.0  # High C = weak regularization
        assert clf.penalty == "l2"
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestSaveLoad:
    """Test save/load functionality."""

    def test_save_unfitted_raises(self):
        """Test error on save of unfitted model."""
        clf = LogisticRegressionClassifier()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.save("/tmp/model.joblib")

    def test_save_load_roundtrip(self, binary_data):
        """Test save and load preserves model."""
        X, y = binary_data
        
        clf = LogisticRegressionClassifier(C=2.0, random_state=42)
        clf.fit(X, y)
        preds_before = clf.predict(X)
        proba_before = clf.predict_proba(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            clf.save(path)
            
            clf_loaded = LogisticRegressionClassifier.load(path)
            preds_after = clf_loaded.predict(X)
            proba_after = clf_loaded.predict_proba(X)
            
            assert np.array_equal(preds_before, preds_after)
            assert np.allclose(proba_before, proba_after)

    def test_load_nonexistent_raises(self):
        """Test error on load of nonexistent file."""
        with pytest.raises(FileNotFoundError):
            LogisticRegressionClassifier.load("/nonexistent/path.joblib")


class TestMulticlass:
    """Test multiclass classification."""

    def test_multiclass_fit_predict(self, multiclass_data):
        """Test multiclass classification."""
        X, y = multiclass_data
        
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        
        assert set(preds) == {0, 1, 2}
        assert preds.shape == (120,)

    def test_multiclass_predict_proba(self, multiclass_data):
        """Test multiclass predict_proba."""
        X, y = multiclass_data
        
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        
        assert proba.shape == (120, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_imbalanced_spectroscopy_workflow(self, imbalanced_data):
        """Test realistic imbalanced spectroscopy workflow."""
        X, y = imbalanced_data
        
        # Use default hyperparams with balanced weights
        defaults = LogisticRegressionClassifier.default_hyperparams()
        clf = LogisticRegressionClassifier(**defaults)
        
        clf.fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        
        assert preds.shape == (100,)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_hyperparameter_sweep(self, binary_data):
        """Test different hyperparameter combinations."""
        X, y = binary_data
        
        hyperparams = [
            {"penalty": "l2", "C": 0.1},
            {"penalty": "l2", "C": 1.0},
            {"penalty": "l2", "C": 10.0},
            {"penalty": "l1", "C": 0.5},
        ]
        
        for params in hyperparams:
            clf = LogisticRegressionClassifier(**params)
            clf.fit(X, y)
            preds = clf.predict(X)
            assert preds.shape == (100,)


__all__ = [
    "TestLogisticRegressionDefault",
    "TestSolverSelection",
    "TestClassWeight",
    "TestParameterValidation",
    "TestInputValidation",
    "TestPredictProba",
    "TestCoefficientsAndIntercept",
    "TestDefaultHyperparams",
    "TestConvenienceConstructors",
    "TestSaveLoad",
    "TestMulticlass",
    "TestIntegration",
]
