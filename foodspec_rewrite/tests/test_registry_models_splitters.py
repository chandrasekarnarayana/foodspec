"""
Tests for Phase 8 - Model and Splitter Registry.

Verifies that all models and splitters can be registered and instantiated:
- Models: logistic_regression, svm, linear_svm, random_forest, pls_da, xgboost, lightgbm
- Splitters: leave_one_batch_out, leave_one_stage_out, leave_one_group_out
- Registry interface: registry.create("model", name, **params)
- Optional dependency handling for XGBoost and LightGBM
"""

import numpy as np
import pandas as pd
import pytest

from foodspec.core.registry import (
    ComponentRegistry,
    register_default_feature_components,
    register_default_model_components,
    register_default_splitter_components,
)


class TestComponentRegistryBasics:
    """Test basic ComponentRegistry functionality."""

    def test_registry_initialization(self):
        """Test ComponentRegistry creates expected categories."""
        registry = ComponentRegistry()
        
        expected_categories = ["preprocess", "qc", "features", "model", "splitter", "plots", "reporters"]
        for category in expected_categories:
            assert category in registry.categories

    def test_register_component(self):
        """Test registering a component."""
        registry = ComponentRegistry()
        
        class DummyComponent:
            def __init__(self, param=1):
                self.param = param
        
        registry.register("model", "dummy", DummyComponent)
        
        assert "dummy" in registry.available("model")

    def test_register_duplicate_raises_error(self):
        """Test registering duplicate component raises ValueError."""
        registry = ComponentRegistry()
        
        class DummyComponent:
            pass
        
        registry.register("model", "dummy", DummyComponent)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register("model", "dummy", DummyComponent)

    def test_register_unknown_category_raises_error(self):
        """Test registering to unknown category raises ValueError."""
        registry = ComponentRegistry()
        
        class DummyComponent:
            pass
        
        with pytest.raises(ValueError, match="Unknown category"):
            registry.register("unknown", "dummy", DummyComponent)

    def test_create_component(self):
        """Test creating a component with parameters."""
        registry = ComponentRegistry()
        
        class DummyComponent:
            def __init__(self, param=1):
                self.param = param
        
        registry.register("model", "dummy", DummyComponent)
        
        instance = registry.create("model", "dummy", param=42)
        assert instance.param == 42

    def test_create_unknown_component_raises_error(self):
        """Test creating unknown component raises ValueError."""
        registry = ComponentRegistry()
        
        with pytest.raises(ValueError, match="Unknown component"):
            registry.create("model", "nonexistent")

    def test_available_lists_components(self):
        """Test available() returns sorted component names."""
        registry = ComponentRegistry()
        
        class Comp1:
            pass
        class Comp2:
            pass
        
        registry.register("model", "comp_b", Comp2)
        registry.register("model", "comp_a", Comp1)
        
        available = registry.available("model")
        assert available == ["comp_a", "comp_b"]  # Sorted


class TestModelRegistration:
    """Test model component registration."""

    def test_register_core_models(self):
        """Test all core models are registered."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        expected_models = [
            "logistic_regression",
            "logreg",
            "svm",
            "linear_svm",
            "random_forest",
            "rf",
            "pls_da",
            "plsda",
        ]
        
        available = registry.available("model")
        for model_name in expected_models:
            assert model_name in available, f"{model_name} not registered"

    def test_xgboost_registered_if_available(self):
        """Test XGBoost is registered if dependency is installed."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        available = registry.available("model")
        
        # XGBoost may or may not be available depending on environment
        try:
            import xgboost
            assert "xgboost" in available
            assert "xgb" in available
        except ImportError:
            assert "xgboost" not in available
            assert "xgb" not in available

    def test_lightgbm_registered_if_available(self):
        """Test LightGBM is registered if dependency is installed."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        available = registry.available("model")
        
        # LightGBM may or may not be available depending on environment
        try:
            import lightgbm
            assert "lightgbm" in available
            assert "lgbm" in available
        except ImportError:
            assert "lightgbm" not in available
            assert "lgbm" not in available

    def test_create_logistic_regression(self):
        """Test creating LogisticRegression model."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "logistic_regression", random_state=42, C=1.0)
        
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
        assert model.random_state == 42
        assert model.C == 1.0

    def test_create_svm(self):
        """Test creating SVM model."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "svm", random_state=42, C=1.0, probability=True)
        
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
        assert model.random_state == 42
        assert model.C == 1.0

    def test_create_linear_svm(self):
        """Test creating Linear SVM model."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "linear_svm", random_state=42, C=1.0)
        
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_create_random_forest(self):
        """Test creating Random Forest model."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "random_forest", random_state=42, n_estimators=100)
        
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
        assert model.random_state == 42
        assert model.n_estimators == 100

    def test_create_pls_da(self):
        """Test creating PLS-DA model."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "pls_da", n_components=5)
        
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "pls")
        assert hasattr(model, "clf")

    def test_create_xgboost(self):
        """Test creating XGBoost model (if available)."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost not installed")
        
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "xgboost", random_state=42, n_estimators=50)
        
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_create_lightgbm(self):
        """Test creating LightGBM model (if available)."""
        try:
            import lightgbm
        except ImportError:
            pytest.skip("lightgbm not installed")
        
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "lightgbm", random_state=42, n_estimators=50)
        
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")


class TestSplitterRegistration:
    """Test splitter component registration."""

    def test_register_splitters(self):
        """Test all splitters are registered."""
        registry = ComponentRegistry()
        register_default_splitter_components(registry)
        
        expected_splitters = [
            "leave_one_group_out",
            "logo",
            "leave_one_batch_out",
            "lobo",
            "leave_one_stage_out",
            "loso",
        ]
        
        available = registry.available("splitter")
        for splitter_name in expected_splitters:
            assert splitter_name in available, f"{splitter_name} not registered"

    def test_create_leave_one_group_out(self):
        """Test creating LeaveOneGroupOut splitter."""
        registry = ComponentRegistry()
        register_default_splitter_components(registry)
        
        splitter = registry.create("splitter", "leave_one_group_out", group_key="batch")
        
        assert hasattr(splitter, "split")
        assert splitter.group_key == "batch"

    def test_create_leave_one_batch_out(self):
        """Test creating LeaveOneBatchOut splitter."""
        registry = ComponentRegistry()
        register_default_splitter_components(registry)
        
        splitter = registry.create("splitter", "leave_one_batch_out")
        
        assert hasattr(splitter, "split")

    def test_create_leave_one_stage_out(self):
        """Test creating LeaveOneStageOut splitter."""
        registry = ComponentRegistry()
        register_default_splitter_components(registry)
        
        splitter = registry.create("splitter", "leave_one_stage_out")
        
        assert hasattr(splitter, "split")


class TestModelFitPredict:
    """Test models can actually fit and predict."""

    def test_logistic_regression_fit_predict(self):
        """Test LogisticRegression can fit and predict."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "logistic_regression", random_state=42)
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        proba = model.predict_proba(X)
        
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_svm_fit_predict(self):
        """Test SVM can fit and predict."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "svm", random_state=42, probability=True)
        
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        proba = model.predict_proba(X)
        
        assert proba.shape == (50, 2)

    def test_random_forest_fit_predict(self):
        """Test RandomForest can fit and predict."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "random_forest", random_state=42, n_estimators=10)
        
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        proba = model.predict_proba(X)
        
        assert proba.shape == (50, 2)

    def test_pls_da_fit_predict(self):
        """Test PLS-DA can fit and predict."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        model = registry.create("model", "pls_da", n_components=3)
        
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        proba = model.predict_proba(X)
        
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestSplitterSplit:
    """Test splitters can actually split data."""

    def test_leave_one_group_out_split(self):
        """Test LeaveOneGroupOut can split data."""
        registry = ComponentRegistry()
        register_default_splitter_components(registry)
        
        splitter = registry.create("splitter", "leave_one_group_out", group_key="batch")
        
        # Generate synthetic data with metadata
        np.random.seed(42)
        X = np.random.randn(30, 10)
        y = np.random.randint(0, 2, 30)
        meta = pd.DataFrame({"batch": [1]*10 + [2]*10 + [3]*10})
        
        folds = list(splitter.split(X, y, meta))
        
        assert len(folds) == 3  # 3 batches
        for train_idx, test_idx, fold_info in folds:
            assert len(train_idx) + len(test_idx) == 30
            assert "held_out_group" in fold_info

    def test_leave_one_batch_out_split(self):
        """Test LeaveOneBatchOut can split data."""
        registry = ComponentRegistry()
        register_default_splitter_components(registry)
        
        splitter = registry.create("splitter", "leave_one_batch_out")
        
        np.random.seed(42)
        X = np.random.randn(30, 10)
        y = np.random.randint(0, 2, 30)
        meta = pd.DataFrame({"batch": [1]*10 + [2]*10 + [3]*10})
        
        folds = list(splitter.split(X, y, meta))
        
        assert len(folds) == 3  # 3 batches

    def test_leave_one_stage_out_split(self):
        """Test LeaveOneStageOut can split data."""
        registry = ComponentRegistry()
        register_default_splitter_components(registry)
        
        splitter = registry.create("splitter", "leave_one_stage_out")
        
        np.random.seed(42)
        X = np.random.randn(30, 10)
        y = np.random.randint(0, 2, 30)
        meta = pd.DataFrame({"stage": ["fresh"]*10 + ["aged_1w"]*10 + ["aged_2w"]*10})
        
        folds = list(splitter.split(X, y, meta))
        
        assert len(folds) == 3  # 3 stages


class TestProtocolV2Integration:
    """Test integration with ProtocolV2 string mappings."""

    def test_protocol_model_strings_map_correctly(self):
        """Test ProtocolV2 model strings map to registered names."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        # Test protocol-style names work
        protocol_names = {
            "logreg": "logistic_regression",
            "rf": "random_forest",
            "xgb": "xgboost",
            "lgbm": "lightgbm",
        }
        
        available = registry.available("model")
        for short_name, full_name in protocol_names.items():
            if short_name in available:  # May not be available (optional deps)
                # Both should work
                model1 = registry.create("model", short_name, random_state=42)
                model2 = registry.create("model", full_name, random_state=42)
                assert type(model1) == type(model2)

    def test_protocol_splitter_strings_map_correctly(self):
        """Test ProtocolV2 splitter strings map to registered names."""
        registry = ComponentRegistry()
        register_default_splitter_components(registry)
        
        # Test protocol-style names work
        protocol_names = {
            "logo": "leave_one_group_out",
            "lobo": "leave_one_batch_out",
            "loso": "leave_one_stage_out",
        }
        
        for short_name, full_name in protocol_names.items():
            # Both should work
            splitter1 = registry.create("splitter", short_name)
            splitter2 = registry.create("splitter", full_name)
            assert type(splitter1) == type(splitter2)


class TestOptionalDependencies:
    """Test graceful handling of optional dependencies."""

    def test_registry_works_without_xgboost(self):
        """Test registry works even if XGBoost not installed."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        # Core models should still be available
        assert "logistic_regression" in registry.available("model")
        assert "random_forest" in registry.available("model")

    def test_registry_works_without_lightgbm(self):
        """Test registry works even if LightGBM not installed."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        # Core models should still be available
        assert "logistic_regression" in registry.available("model")
        assert "random_forest" in registry.available("model")

    def test_error_message_for_missing_optional_model(self):
        """Test clear error message when trying to create unavailable optional model."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        
        # Try to create a model that might not be available
        try:
            import xgboost
            xgboost_available = True
        except ImportError:
            xgboost_available = False
        
        if not xgboost_available:
            with pytest.raises(ValueError, match="Unknown component.*xgboost"):
                registry.create("model", "xgboost")


class TestEndToEndWorkflow:
    """Test complete workflow with registry."""

    def test_full_workflow_model_and_splitter(self):
        """Test creating model and splitter, then using in CV."""
        registry = ComponentRegistry()
        register_default_model_components(registry)
        register_default_splitter_components(registry)
        
        # Create model and splitter from registry
        model = registry.create("model", "logistic_regression", random_state=42)
        splitter = registry.create("splitter", "leave_one_group_out", group_key="batch")
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(30, 10)
        y = np.random.randint(0, 2, 30)
        meta = pd.DataFrame({"batch": [1]*10 + [2]*10 + [3]*10})
        
        # Run simple CV loop
        accuracies = []
        for train_idx, test_idx, fold_info in splitter.split(X, y, meta):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone model for each fold
            import copy
            model_fold = copy.deepcopy(model)
            model_fold.fit(X_train, y_train)
            
            proba = model_fold.predict_proba(X_test)
            pred = proba.argmax(axis=1)
            acc = (pred == y_test).mean()
            accuracies.append(acc)
        
        assert len(accuracies) == 3  # 3 folds
        assert all(0 <= acc <= 1 for acc in accuracies)
