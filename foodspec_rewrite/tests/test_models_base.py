"""
Unit tests for base model interface and label encoder wrapper.

Tests cover:
- LabelEncoderWrapper: fit/transform, inverse_transform, class consistency
- BaseEstimator interface: fit/predict, parameter management, cloning, save/load
- Model interchangeability: ensuring models with same interface work in evaluation
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from foodspec.models.base import BaseEstimator, LabelEncoderWrapper


class TestLabelEncoderWrapper:
    """Test LabelEncoderWrapper for consistent label encoding."""

    def test_fit_transform_numeric_labels(self):
        """Test encoding of numeric labels."""
        encoder = LabelEncoderWrapper()
        y = np.array([2, 0, 1, 2, 0])
        
        y_encoded = encoder.fit_transform(y)
        
        # Should produce consistent numeric encoding
        assert y_encoded.dtype == np.int64
        assert len(encoder.classes_) == 3
        assert np.array_equal(np.unique(y_encoded), [0, 1, 2])

    def test_fit_transform_string_labels(self):
        """Test encoding of string labels."""
        encoder = LabelEncoderWrapper()
        y = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])
        
        y_encoded = encoder.fit_transform(y)
        
        # Should encode strings to integers
        assert y_encoded.dtype == np.int64
        assert len(encoder.classes_) == 3
        # Classes should be sorted alphabetically
        assert list(encoder.classes_) == ['bird', 'cat', 'dog']

    def test_transform_unseen_classes(self):
        """Test transform raises error on unseen classes."""
        encoder = LabelEncoderWrapper()
        y = np.array([0, 1, 2])
        encoder.fit_transform(y)
        
        # Transform with unseen class should raise error
        with pytest.raises(ValueError):
            encoder.transform(np.array([0, 1, 99]))

    def test_inverse_transform(self):
        """Test reverse transformation to original labels."""
        encoder = LabelEncoderWrapper()
        y_original = np.array(['cat', 'dog', 'cat', 'bird'])
        
        y_encoded = encoder.fit_transform(y_original)
        y_reconstructed = encoder.inverse_transform(y_encoded)
        
        assert np.array_equal(y_original, y_reconstructed)

    def test_n_classes_property(self):
        """Test n_classes property."""
        encoder = LabelEncoderWrapper()
        y = np.array([0, 1, 2, 0, 1])
        encoder.fit_transform(y)
        
        assert encoder.n_classes == 3

    def test_unfitted_transform_raises(self):
        """Test that transform on unfitted encoder raises error."""
        encoder = LabelEncoderWrapper()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            encoder.transform(np.array([0, 1, 2]))

    def test_unfitted_inverse_raises(self):
        """Test that inverse_transform on unfitted encoder raises error."""
        encoder = LabelEncoderWrapper()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            encoder.inverse_transform(np.array([0, 1, 2]))

    def test_unfitted_classes_raises(self):
        """Test that accessing classes_ on unfitted encoder raises error."""
        encoder = LabelEncoderWrapper()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = encoder.classes_

    def test_binary_classification(self):
        """Test binary classification encoding."""
        encoder = LabelEncoderWrapper()
        y = np.array([0, 1, 0, 1, 0])
        
        y_encoded = encoder.fit_transform(y)
        
        assert encoder.n_classes == 2
        assert np.array_equal(y, y_encoded)

    def test_multiclass_classification(self):
        """Test multiclass classification encoding."""
        encoder = LabelEncoderWrapper()
        y = np.array([5, 3, 8, 3, 5, 8])
        
        y_encoded = encoder.fit_transform(y)
        
        # Should map to 0, 1, 2
        assert encoder.n_classes == 3
        assert set(y_encoded) == {0, 1, 2}

    def test_invalid_y_shape_fit(self):
        """Test that 2D y raises error."""
        encoder = LabelEncoderWrapper()
        
        with pytest.raises(ValueError, match="1D"):
            encoder.fit_transform(np.array([[0, 1], [1, 0]]))

    def test_invalid_y_shape_transform(self):
        """Test that 2D y in transform raises error."""
        encoder = LabelEncoderWrapper()
        encoder.fit_transform(np.array([0, 1, 0]))
        
        with pytest.raises(ValueError, match="1D"):
            encoder.transform(np.array([[0, 1]]))


class DummyEstimator(BaseEstimator):
    """Simple test implementation of BaseEstimator."""

    def __init__(self, alpha: float = 1.0, beta: int = 10) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self._trained_mean = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyEstimator":
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same length")
        self._trained_mean = np.mean(X, axis=0)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X = np.asarray(X)
        # Simple dummy prediction based on feature mean
        distances = np.sum((X - self._trained_mean) ** 2, axis=1)
        return (distances > np.median(distances)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X = np.asarray(X)
        predictions = self.predict(X)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, 2))
        proba[:, 1] = predictions
        proba[:, 0] = 1 - predictions
        return proba


class TestBaseEstimatorInterface:
    """Test BaseEstimator interface methods."""

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        model = DummyEstimator()
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        
        result = model.fit(X, y)
        
        assert result is model

    def test_predict_requires_fit(self):
        """Test that predict raises error on unfitted model."""
        model = DummyEstimator()
        X = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_predict_proba_requires_fit(self):
        """Test that predict_proba raises error on unfitted model."""
        model = DummyEstimator()
        X = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X)

    def test_predict_shape(self):
        """Test predict output shape."""
        model = DummyEstimator()
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        model.fit(X, y)
        
        preds = model.predict(X)
        
        assert preds.shape == (10,)

    def test_predict_proba_shape(self):
        """Test predict_proba output shape."""
        model = DummyEstimator()
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_get_params(self):
        """Test get_params returns all public attributes."""
        model = DummyEstimator(alpha=2.5, beta=20)
        
        params = model.get_params()
        
        assert params['alpha'] == 2.5
        assert params['beta'] == 20
        assert '_fitted' not in params
        assert '_trained_mean' not in params

    def test_set_params(self):
        """Test set_params updates parameters."""
        model = DummyEstimator(alpha=1.0)
        
        model.set_params(alpha=5.0, beta=50)
        
        assert model.alpha == 5.0
        assert model.beta == 50

    def test_set_params_returns_self(self):
        """Test set_params returns self for method chaining."""
        model = DummyEstimator()
        
        result = model.set_params(alpha=2.0)
        
        assert result is model

    def test_set_params_invalid_parameter(self):
        """Test set_params raises error on invalid parameter."""
        model = DummyEstimator()
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(invalid_param=99)

    def test_set_params_empty_dict(self):
        """Test set_params with empty params."""
        model = DummyEstimator(alpha=1.0)
        
        result = model.set_params()
        
        assert result is model
        assert model.alpha == 1.0

    def test_clone_with_params(self):
        """Test clone_with_params creates independent copy."""
        model1 = DummyEstimator(alpha=1.0)
        model2 = model1.clone_with_params(alpha=5.0)
        
        # Original unchanged
        assert model1.alpha == 1.0
        # New model has updated params
        assert model2.alpha == 5.0
        # Different instances
        assert model1 is not model2

    def test_clone_with_params_preserves_fit_state(self):
        """Test clone_with_params copies fitted state."""
        model = DummyEstimator()
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        model.fit(X, y)
        
        cloned = model.clone_with_params(alpha=2.0)
        
        # Cloned model should also be fitted
        assert cloned._fitted
        assert cloned._trained_mean is not None

    def test_clone_with_params_independent(self):
        """Test that cloned model is independent from original."""
        model1 = DummyEstimator()
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        model1.fit(X, y)
        
        model2 = model1.clone_with_params(alpha=2.0)
        model2.alpha = 10.0
        
        # Original should not be affected
        assert model1.alpha == 1.0

    def test_save_unfitted_raises(self):
        """Test that save on unfitted model raises error."""
        model = DummyEstimator()
        
        with pytest.raises(RuntimeError, match="unfitted"):
            model.save("/tmp/model.joblib")

    def test_save_and_load_roundtrip(self):
        """Test save/load roundtrip preserves model state."""
        model = DummyEstimator(alpha=2.5, beta=15)
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            model.save(path)
            
            loaded = DummyEstimator.load(path)
            
            # Parameters preserved
            assert loaded.alpha == 2.5
            assert loaded.beta == 15
            # Fitted state preserved
            assert loaded._fitted
            # Predictions identical
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            assert np.array_equal(preds_orig, preds_loaded)

    def test_load_missing_file(self):
        """Test load raises error on missing file."""
        with pytest.raises(FileNotFoundError):
            DummyEstimator.load("/nonexistent/path/model.joblib")

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories."""
        model = DummyEstimator()
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "model.joblib"
            model.save(path)
            
            assert path.exists()

    def test_predict_proba_unsupported_model(self):
        """Test predict_proba raises clear error on unsupported model."""
        
        class NoProbaEstimator(BaseEstimator):
            def fit(self, X, y):
                self._fitted = True
                return self
            
            def predict(self, X):
                return np.zeros(X.shape[0])
        
        model = NoProbaEstimator()
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        model.fit(X, y)
        
        with pytest.raises(NotImplementedError, match="does not support predict_proba"):
            model.predict_proba(X)


class TestModelInterchangeability:
    """Test that models with same interface are interchangeable."""

    def test_multiple_models_same_interface(self):
        """Test that multiple models follow same interface."""
        models = [
            DummyEstimator(alpha=1.0),
            DummyEstimator(alpha=2.0),
        ]
        
        X = np.random.randn(20, 5)
        y = np.array([0, 1] * 10)
        
        # All models should work with same code
        for model in models:
            model.fit(X, y)
            preds = model.predict(X)
            proba = model.predict_proba(X)
            
            assert preds.shape == (20,)
            assert proba.shape == (20, 2)

    def test_model_as_function_parameter(self):
        """Test that models can be passed as function parameter."""
        
        def evaluate_model(estimator, X_train, y_train, X_test, y_test):
            """Generic evaluation function accepting any BaseEstimator."""
            estimator.fit(X_train, y_train)
            train_preds = estimator.predict(X_train)
            test_preds = estimator.predict(X_test)
            
            train_acc = np.mean(train_preds == y_train)
            test_acc = np.mean(test_preds == y_test)
            
            return train_acc, test_acc
        
        model = DummyEstimator()
        X_train = np.random.randn(20, 5)
        y_train = np.array([0, 1] * 10)
        X_test = np.random.randn(10, 5)
        y_test = np.array([0, 1] * 5)
        
        train_acc, test_acc = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        assert 0 <= train_acc <= 1
        assert 0 <= test_acc <= 1

    def test_parameter_sweep(self):
        """Test parameter sweep without changing evaluation code."""
        
        def get_best_model(candidates, X_train, y_train):
            """Select best model from candidates."""
            best_model = None
            best_score = -1
            
            for model in candidates:
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                score = np.mean(train_preds == y_train)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            return best_model
        
        candidates = [
            DummyEstimator(alpha=0.5),
            DummyEstimator(alpha=1.0),
            DummyEstimator(alpha=2.0),
        ]
        
        X = np.random.randn(20, 5)
        y = np.array([0, 1] * 10)
        
        best = get_best_model(candidates, X, y)
        assert best is not None


__all__ = ["TestLabelEncoderWrapper", "TestBaseEstimatorInterface", "TestModelInterchangeability"]
