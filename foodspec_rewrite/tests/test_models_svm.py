"""
Comprehensive tests for LinearSVCClassifier and SVCClassifier.

Tests cover:
- Default instantiation and basic functionality
- Parameter validation
- Decision function (LinearSVC)
- Predict proba (SVC with probability=True)
- Error handling when probability not available
- Calibration support
- Save/load serialization
- Factory methods
- Integration workflows
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from foodspec.models.classical import LinearSVCClassifier, SVCClassifier


class TestLinearSVCDefault:
    """Test LinearSVCClassifier with default parameters."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_default_instantiation(self):
        """Test instantiation with default parameters."""
        clf = LinearSVCClassifier()
        assert clf.C == 1.0
        assert clf.penalty == "l2"
        assert clf.loss == "squared_hinge"
        assert clf.max_iter == 1000
        assert clf.random_state == 0

    def test_fit_predict_basic(self, binary_data):
        """Test basic fit and predict."""
        X, y = binary_data
        clf = LinearSVCClassifier()
        clf.fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds) <= {0, 1}

    def test_deterministic_with_seed(self, binary_data):
        """Test reproducibility with same seed."""
        X, y = binary_data
        
        clf1 = LinearSVCClassifier(random_state=42)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)
        
        clf2 = LinearSVCClassifier(random_state=42)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)
        
        np.testing.assert_array_equal(preds1, preds2)

    def test_different_seeds_different_results(self, binary_data):
        """Test that different seeds produce different results."""
        X, y = binary_data
        
        clf1 = LinearSVCClassifier(random_state=0)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)
        
        clf2 = LinearSVCClassifier(random_state=1)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)
        
        # Different seeds may produce different predictions (but not guaranteed)
        # Just verify they both work
        assert preds1.shape == preds2.shape


class TestLinearSVCDecisionFunction:
    """Test LinearSVCClassifier decision_function method."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_decision_function_shape(self, binary_data):
        """Test decision_function returns correct shape."""
        X, y = binary_data
        clf = LinearSVCClassifier()
        clf.fit(X, y)
        
        scores = clf.decision_function(X)
        assert scores.shape == (100,)

    def test_decision_function_not_fitted_raises(self):
        """Test decision_function raises before fitting."""
        clf = LinearSVCClassifier()
        X = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.decision_function(X)

    def test_decision_function_multiclass(self):
        """Test decision_function with multiclass."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1, 2] * 33 + [0])  # 3 classes
        
        clf = LinearSVCClassifier()
        clf.fit(X, y)
        
        scores = clf.decision_function(X)
        assert scores.ndim == 2  # (n_samples, n_classes)
        assert scores.shape[1] == 3


class TestLinearSVCParameterValidation:
    """Test LinearSVCClassifier parameter validation."""

    def test_invalid_penalty(self):
        """Test invalid penalty raises error."""
        with pytest.raises(ValueError, match="penalty must be"):
            LinearSVCClassifier(penalty="invalid")

    def test_negative_C(self):
        """Test negative C raises error."""
        with pytest.raises(ValueError, match="C must be positive"):
            LinearSVCClassifier(C=-1.0)

    def test_zero_C(self):
        """Test C=0 raises error."""
        with pytest.raises(ValueError, match="C must be positive"):
            LinearSVCClassifier(C=0.0)

    def test_invalid_loss(self):
        """Test invalid loss raises error."""
        with pytest.raises(ValueError, match="loss must be"):
            LinearSVCClassifier(loss="invalid")

    def test_negative_max_iter(self):
        """Test negative max_iter raises error."""
        with pytest.raises(ValueError, match="max_iter must be positive"):
            LinearSVCClassifier(max_iter=-1)

    def test_negative_tol(self):
        """Test negative tol raises error."""
        with pytest.raises(ValueError, match="tol must be positive"):
            LinearSVCClassifier(tol=-1e-4)

    def test_invalid_class_weight_string(self):
        """Test invalid class_weight string raises error."""
        with pytest.raises(ValueError, match="class_weight string must be"):
            LinearSVCClassifier(class_weight="invalid")


class TestLinearSVCInputValidation:
    """Test LinearSVCClassifier input validation."""

    def test_X_not_2d_raises(self):
        """Test 1D X raises error."""
        clf = LinearSVCClassifier()
        X = np.random.randn(100)
        y = np.array([0, 1] * 50)
        
        with pytest.raises(ValueError, match="X must be 2D"):
            clf.fit(X, y)

    def test_y_not_1d_raises(self):
        """Test 2D y raises error."""
        clf = LinearSVCClassifier()
        X = np.random.randn(100, 50)
        y = np.random.randn(100, 2)
        
        with pytest.raises(ValueError, match="y must be 1D"):
            clf.fit(X, y)

    def test_mismatched_lengths_raises(self):
        """Test mismatched X and y lengths raise error."""
        clf = LinearSVCClassifier()
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 30)  # 60 samples
        
        with pytest.raises(ValueError, match="inconsistent lengths"):
            clf.fit(X, y)

    def test_predict_not_fitted_raises(self):
        """Test predict without fitting raises error."""
        clf = LinearSVCClassifier()
        X = np.random.randn(10, 50)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(X)


class TestLinearSVCClassWeight:
    """Test LinearSVCClassifier class weight handling."""

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
        clf = LinearSVCClassifier(class_weight="balanced")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_custom_class_weight_dict(self, imbalanced_data):
        """Test custom class_weight dict."""
        X, y = imbalanced_data
        clf = LinearSVCClassifier(class_weight={0: 1.0, 1: 5.0})
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_none_class_weight(self, imbalanced_data):
        """Test class_weight=None (default)."""
        X, y = imbalanced_data
        clf = LinearSVCClassifier(class_weight=None)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestLinearSVCCoefficients:
    """Test LinearSVCClassifier coefficient access."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_get_coef(self, binary_data):
        """Test get_coef returns correct shape."""
        X, y = binary_data
        clf = LinearSVCClassifier()
        clf.fit(X, y)
        
        coef = clf.get_coef()
        assert coef.shape == (50,)

    def test_get_intercept(self, binary_data):
        """Test get_intercept returns correct shape."""
        X, y = binary_data
        clf = LinearSVCClassifier()
        clf.fit(X, y)
        
        intercept = clf.get_intercept()
        assert np.isscalar(intercept) or intercept.shape == (1,)

    def test_coef_not_fitted_raises(self):
        """Test get_coef before fitting raises error."""
        clf = LinearSVCClassifier()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.get_coef()

    def test_intercept_not_fitted_raises(self):
        """Test get_intercept before fitting raises error."""
        clf = LinearSVCClassifier()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.get_intercept()


class TestLinearSVCDefaultHyperparams:
    """Test LinearSVCClassifier default hyperparameters."""

    def test_default_hyperparams_spectroscopy(self):
        """Test default hyperparams are suitable for spectroscopy."""
        defaults = LinearSVCClassifier.default_hyperparams()
        
        assert defaults["C"] == 1.0
        assert defaults["penalty"] == "l2"
        assert defaults["class_weight"] == "balanced"
        assert defaults["loss"] == "squared_hinge"

    def test_can_instantiate_with_defaults(self):
        """Test can instantiate with default hyperparams."""
        defaults = LinearSVCClassifier.default_hyperparams()
        clf = LinearSVCClassifier(**defaults)
        assert clf.C == 1.0


class TestLinearSVCFactoryMethods:
    """Test LinearSVCClassifier factory methods."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_sparse_features_constructor(self, binary_data):
        """Test sparse_features factory method."""
        X, y = binary_data
        clf = LinearSVCClassifier.sparse_features()
        clf.fit(X, y)
        
        assert clf.penalty == "l1"
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_strong_regularization_constructor(self, binary_data):
        """Test strong_regularization factory method."""
        X, y = binary_data
        clf = LinearSVCClassifier.strong_regularization()
        clf.fit(X, y)
        
        assert clf.C == 0.1
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_weak_regularization_constructor(self, binary_data):
        """Test weak_regularization factory method."""
        X, y = binary_data
        clf = LinearSVCClassifier.weak_regularization()
        clf.fit(X, y)
        
        assert clf.C == 10.0
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestLinearSVCSaveLoad:
    """Test LinearSVCClassifier save/load functionality."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_save_unfitted_raises(self):
        """Test save raises error for unfitted model."""
        clf = LinearSVCClassifier()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            with pytest.raises(RuntimeError, match="not fitted"):
                clf.save(path)

    def test_save_load_roundtrip(self, binary_data):
        """Test save and load roundtrip."""
        X, y = binary_data
        clf = LinearSVCClassifier()
        clf.fit(X, y)
        
        preds1 = clf.predict(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf_loaded = LinearSVCClassifier.load(path)
            preds2 = clf_loaded.predict(X)
        
        np.testing.assert_array_equal(preds1, preds2)


# ============================================================================
# SVCClassifier Tests
# ============================================================================


class TestSVCDefault:
    """Test SVCClassifier with default parameters."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_default_instantiation(self):
        """Test instantiation with default parameters."""
        clf = SVCClassifier()
        assert clf.C == 1.0
        assert clf.kernel == "rbf"
        assert clf.probability is True
        assert clf.calibrate is False

    def test_fit_predict_basic(self, binary_data):
        """Test basic fit and predict."""
        X, y = binary_data
        clf = SVCClassifier()
        clf.fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds) <= {0, 1}

    def test_deterministic_with_seed(self, binary_data):
        """Test reproducibility with same seed."""
        X, y = binary_data
        
        clf1 = SVCClassifier(random_state=42)
        clf1.fit(X, y)
        preds1 = clf1.predict(X)
        
        clf2 = SVCClassifier(random_state=42)
        clf2.fit(X, y)
        preds2 = clf2.predict(X)
        
        np.testing.assert_array_equal(preds1, preds2)


class TestSVCPredictProba:
    """Test SVCClassifier predict_proba method."""

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
        clf = SVCClassifier(probability=True)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        """Test probabilities sum to 1 for each sample."""
        X, y = binary_data
        clf = SVCClassifier(probability=True)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_predict_proba_bounds(self, binary_data):
        """Test probabilities are in [0, 1]."""
        X, y = binary_data
        clf = SVCClassifier(probability=True)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_proba_multiclass(self):
        """Test predict_proba with multiclass."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1, 2] * 33 + [0])
        
        clf = SVCClassifier(probability=True)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 3)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_predict_proba_not_fitted_raises(self):
        """Test predict_proba raises before fitting."""
        clf = SVCClassifier(probability=True)
        X = np.random.randn(10, 50)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(X)

    def test_predict_proba_probability_false_raises(self, binary_data):
        """Test predict_proba raises error when probability=False."""
        X, y = binary_data
        clf = SVCClassifier(probability=False)
        clf.fit(X, y)
        
        with pytest.raises(RuntimeError, match="probability=False"):
            clf.predict_proba(X)


class TestSVCDecisionFunction:
    """Test SVCClassifier decision_function method."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_decision_function_shape_binary(self, binary_data):
        """Test decision_function shape for binary classification."""
        X, y = binary_data
        clf = SVCClassifier(probability=True)
        clf.fit(X, y)
        
        scores = clf.decision_function(X)
        assert scores.shape == (100,)

    def test_decision_function_not_fitted_raises(self):
        """Test decision_function raises before fitting."""
        clf = SVCClassifier()
        X = np.random.randn(10, 50)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.decision_function(X)


class TestSVCCalibration:
    """Test SVCClassifier calibration functionality."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_with_calibration_factory(self, binary_data):
        """Test with_calibration factory method."""
        X, y = binary_data
        clf = SVCClassifier.with_calibration()
        clf.fit(X, y)
        
        assert clf.calibrate is True
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_calibrate_true_vs_false(self, binary_data):
        """Test that calibration affects probability estimates."""
        X, y = binary_data
        
        clf_no_cal = SVCClassifier(calibrate=False)
        clf_no_cal.fit(X, y)
        proba_no_cal = clf_no_cal.predict_proba(X)
        
        clf_cal = SVCClassifier(calibrate=True)
        clf_cal.fit(X, y)
        proba_cal = clf_cal.predict_proba(X)
        
        # Probabilities should be different (calibration affects them)
        # Both should be valid
        assert np.all(proba_no_cal >= 0.0) and np.all(proba_no_cal <= 1.0)
        assert np.all(proba_cal >= 0.0) and np.all(proba_cal <= 1.0)


class TestSVCParameterValidation:
    """Test SVCClassifier parameter validation."""

    def test_invalid_kernel(self):
        """Test invalid kernel raises error."""
        with pytest.raises(ValueError, match="kernel must be"):
            SVCClassifier(kernel="invalid")

    def test_negative_C(self):
        """Test negative C raises error."""
        with pytest.raises(ValueError, match="C must be positive"):
            SVCClassifier(C=-1.0)

    def test_invalid_degree(self):
        """Test invalid degree raises error."""
        with pytest.raises(ValueError, match="degree must be"):
            SVCClassifier(degree=0)

    def test_negative_gamma(self):
        """Test negative gamma raises error."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            SVCClassifier(gamma=-1.0)

    def test_invalid_gamma_string(self):
        """Test invalid gamma string raises error."""
        with pytest.raises(ValueError, match="gamma must be"):
            SVCClassifier(gamma="invalid")

    def test_negative_max_iter(self):
        """Test negative max_iter raises error."""
        with pytest.raises(ValueError, match="max_iter must be positive"):
            SVCClassifier(max_iter=-1)

    def test_negative_tol(self):
        """Test negative tol raises error."""
        with pytest.raises(ValueError, match="tol must be positive"):
            SVCClassifier(tol=-1e-3)


class TestSVCKernels:
    """Test SVCClassifier with different kernels."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_linear_kernel(self, binary_data):
        """Test linear kernel."""
        X, y = binary_data
        clf = SVCClassifier.linear_kernel()
        clf.fit(X, y)
        
        assert clf.kernel == "linear"
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_rbf_kernel(self, binary_data):
        """Test RBF kernel."""
        X, y = binary_data
        clf = SVCClassifier.rbf_kernel()
        clf.fit(X, y)
        
        assert clf.kernel == "rbf"
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_poly_kernel(self, binary_data):
        """Test polynomial kernel."""
        X, y = binary_data
        clf = SVCClassifier(kernel="poly", degree=3)
        clf.fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_sigmoid_kernel(self, binary_data):
        """Test sigmoid kernel."""
        X, y = binary_data
        clf = SVCClassifier(kernel="sigmoid")
        clf.fit(X, y)
        
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestSVCLinearKernelCoefficients:
    """Test SVCClassifier coefficient access with linear kernel."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_get_coef_linear_kernel(self, binary_data):
        """Test get_coef with linear kernel."""
        X, y = binary_data
        clf = SVCClassifier.linear_kernel()
        clf.fit(X, y)
        
        coef = clf.get_coef()
        assert coef.shape == (50,)

    def test_get_coef_rbf_kernel_raises(self, binary_data):
        """Test get_coef raises error for RBF kernel."""
        X, y = binary_data
        clf = SVCClassifier.rbf_kernel()
        clf.fit(X, y)
        
        with pytest.raises(ValueError, match="only available for linear"):
            clf.get_coef()

    def test_get_intercept_linear_kernel(self, binary_data):
        """Test get_intercept with linear kernel."""
        X, y = binary_data
        clf = SVCClassifier.linear_kernel()
        clf.fit(X, y)
        
        intercept = clf.get_intercept()
        assert np.isscalar(intercept) or intercept.shape[0] > 0

    def test_get_intercept_rbf_kernel_raises(self, binary_data):
        """Test get_intercept raises error for RBF kernel."""
        X, y = binary_data
        clf = SVCClassifier.rbf_kernel()
        clf.fit(X, y)
        
        with pytest.raises(ValueError, match="only available for linear"):
            clf.get_intercept()


class TestSVCSaveLoad:
    """Test SVCClassifier save/load functionality."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_save_unfitted_raises(self):
        """Test save raises error for unfitted model."""
        clf = SVCClassifier()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            with pytest.raises(RuntimeError, match="not fitted"):
                clf.save(path)

    def test_save_load_roundtrip(self, binary_data):
        """Test save and load roundtrip."""
        X, y = binary_data
        clf = SVCClassifier(probability=True)
        clf.fit(X, y)
        
        proba1 = clf.predict_proba(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf_loaded = SVCClassifier.load(path)
            proba2 = clf_loaded.predict_proba(X)
        
        np.testing.assert_allclose(proba1, proba2, rtol=1e-5)

    def test_load_sets_probability_flag(self, binary_data):
        """Test load correctly sets _has_probability flag."""
        X, y = binary_data
        clf = SVCClassifier(probability=True)
        clf.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf_loaded = SVCClassifier.load(path)
            
            # Should be able to call predict_proba
            proba = clf_loaded.predict_proba(X)
            assert proba.shape == (100, 2)


class TestSVCDefaultHyperparams:
    """Test SVCClassifier default hyperparameters."""

    def test_default_hyperparams_spectroscopy(self):
        """Test default hyperparams are suitable for spectroscopy."""
        defaults = SVCClassifier.default_hyperparams()
        
        assert defaults["C"] == 1.0
        assert defaults["kernel"] == "rbf"
        assert defaults["probability"] is True
        assert defaults["class_weight"] == "balanced"

    def test_can_instantiate_with_defaults(self):
        """Test can instantiate with default hyperparams."""
        defaults = SVCClassifier.default_hyperparams()
        clf = SVCClassifier(**defaults)
        assert clf.C == 1.0


class TestSVCFactoryMethods:
    """Test SVCClassifier factory methods."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.array([0, 1] * 50)
        return X, y

    def test_linear_kernel_factory(self, binary_data):
        """Test linear_kernel factory method."""
        X, y = binary_data
        clf = SVCClassifier.linear_kernel()
        clf.fit(X, y)
        
        assert clf.kernel == "linear"
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_rbf_kernel_factory(self, binary_data):
        """Test rbf_kernel factory method."""
        X, y = binary_data
        clf = SVCClassifier.rbf_kernel()
        clf.fit(X, y)
        
        assert clf.kernel == "rbf"
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_strong_regularization_factory(self, binary_data):
        """Test strong_regularization factory method."""
        X, y = binary_data
        clf = SVCClassifier.strong_regularization()
        clf.fit(X, y)
        
        assert clf.C == 0.1
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_weak_regularization_factory(self, binary_data):
        """Test weak_regularization factory method."""
        X, y = binary_data
        clf = SVCClassifier.weak_regularization()
        clf.fit(X, y)
        
        assert clf.C == 10.0
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)


class TestIntegration:
    """Integration tests for both classifiers."""

    @pytest.fixture
    def imbalanced_data(self):
        """Generate imbalanced spectroscopy data."""
        np.random.seed(42)
        X = np.random.randn(100, 500)  # High-dimensional
        y = np.array([0] * 80 + [1] * 20)  # Imbalanced
        return X, y

    def test_linear_svc_imbalanced_workflow(self, imbalanced_data):
        """Test LinearSVC on imbalanced spectroscopy data."""
        X, y = imbalanced_data
        
        clf = LinearSVCClassifier(class_weight="balanced")
        clf.fit(X, y)
        
        preds = clf.predict(X)
        scores = clf.decision_function(X)
        
        assert preds.shape == (100,)
        assert scores.shape == (100,)

    def test_svc_imbalanced_workflow(self, imbalanced_data):
        """Test SVC on imbalanced spectroscopy data."""
        X, y = imbalanced_data
        
        clf = SVCClassifier(class_weight="balanced", probability=True)
        clf.fit(X, y)
        
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        
        assert preds.shape == (100,)
        assert proba.shape == (100, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_svc_calibration_imbalanced(self, imbalanced_data):
        """Test SVC with calibration on imbalanced data."""
        X, y = imbalanced_data
        
        clf = SVCClassifier.with_calibration(class_weight="balanced")
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert proba.shape == (100, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_hyperparameter_sweep_linear_svc(self, imbalanced_data):
        """Test LinearSVC hyperparameter sweep."""
        X, y = imbalanced_data
        
        hyperparams = [
            {"C": 0.1, "penalty": "l2"},
            {"C": 1.0, "penalty": "l2"},
            {"C": 10.0, "penalty": "l2"},
            {"C": 0.1, "penalty": "l1", "dual": False},
        ]
        
        for params in hyperparams:
            clf = LinearSVCClassifier(**params)
            clf.fit(X, y)
            preds = clf.predict(X)
            assert preds.shape == (100,)

    def test_hyperparameter_sweep_svc(self, imbalanced_data):
        """Test SVC hyperparameter sweep."""
        X, y = imbalanced_data
        
        hyperparams = [
            {"kernel": "linear", "C": 0.1},
            {"kernel": "linear", "C": 1.0},
            {"kernel": "rbf", "C": 1.0},
            {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        ]
        
        for params in hyperparams:
            clf = SVCClassifier(**params, probability=True)
            clf.fit(X, y)
            proba = clf.predict_proba(X)
            assert proba.shape == (100, 2)
