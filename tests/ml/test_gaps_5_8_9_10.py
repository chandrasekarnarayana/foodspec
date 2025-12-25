"""
Unit tests for gaps 5, 8, 9, 10 implementations.

Tests threshold optimization, hyperparameter tuning, nested CV, and memory management.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Import implemented modules
from foodspec.qc.threshold_optimization import (
    estimate_threshold_quantile,
    estimate_threshold_youden,
    estimate_threshold_f1,
    estimate_threshold_elbow,
    validate_threshold,
)

from foodspec.ml.hyperparameter_tuning import (
    get_search_space_classifier,
    get_search_space_regressor,
    grid_search_classifier,
    grid_search_regressor,
    quick_tune_classifier,
)

from foodspec.ml.nested_cv import (
    nested_cross_validate,
    nested_cross_validate_regression,
    compare_models_nested_cv,
)

from foodspec.hyperspectral.memory_management import (
    HyperspectralStreamReader,
    HyperspectralTiler,
    process_hyperspectral_chunks,
    estimate_memory_usage,
    recommend_chunk_size,
)


# ============================================================================
# GAP 5: Threshold Optimization Tests
# ============================================================================

class TestThresholdOptimization:
    """Test automated threshold tuning for QC metrics."""

    @pytest.fixture
    def synthetic_scores(self):
        """Generate synthetic anomaly scores (normal=low, anomalous=high)."""
        normal = np.random.normal(0.3, 0.1, 100)
        anomalous = np.random.normal(0.7, 0.1, 100)
        scores = np.concatenate([normal, anomalous])
        labels = np.concatenate([np.zeros(100), np.ones(100)])
        return scores, labels

    def test_quantile_threshold(self, synthetic_scores):
        """Test quantile-based threshold estimation."""
        scores, _ = synthetic_scores
        threshold = estimate_threshold_quantile(scores, percentile=95)
        assert isinstance(threshold, (float, np.floating))
        assert 0.0 <= threshold <= 1.0
        # Should be ~0.95 of max for normal distribution
        assert threshold > np.percentile(scores, 50)

    def test_youden_threshold(self, synthetic_scores):
        """Test Youden's J-statistic optimization."""
        scores, labels = synthetic_scores
        threshold = estimate_threshold_youden(scores, labels)
        assert isinstance(threshold, (float, np.floating))
        # Threshold should be between min and max scores
        assert scores.min() <= threshold <= scores.max()

    def test_f1_threshold(self, synthetic_scores):
        """Test F1-score maximization."""
        scores, labels = synthetic_scores
        threshold = estimate_threshold_f1(scores, labels)
        assert isinstance(threshold, (float, np.floating))
        assert scores.min() <= threshold <= scores.max()

    def test_elbow_threshold(self, synthetic_scores):
        """Test unsupervised elbow detection."""
        scores, _ = synthetic_scores
        threshold = estimate_threshold_elbow(scores, n_clusters=2)
        assert isinstance(threshold, (float, np.floating))
        # For bimodal distribution, elbow should separate modes
        assert scores.min() < threshold < scores.max()

    def test_threshold_validation(self, synthetic_scores):
        """Test threshold validation metrics."""
        scores, labels = synthetic_scores
        threshold = np.median(scores)
        metrics = validate_threshold(scores, labels, threshold)
        
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "precision" in metrics
        assert "f1_score" in metrics
        assert "accuracy" in metrics
        
        for key in metrics:
            assert 0.0 <= metrics[key] <= 1.0

    def test_threshold_methods_consistency(self, synthetic_scores):
        """Test that all methods produce reasonable thresholds."""
        scores, labels = synthetic_scores
        
        t_quantile = estimate_threshold_quantile(scores, percentile=90)
        t_youden = estimate_threshold_youden(scores, labels)
        t_f1 = estimate_threshold_f1(scores, labels)
        t_elbow = estimate_threshold_elbow(scores, n_clusters=2)
        
        # All thresholds should be in reasonable range
        thresholds = [t_quantile, t_youden, t_f1, t_elbow]
        for t in thresholds:
            assert scores.min() <= t <= scores.max()


# ============================================================================
# GAP 8: Hyperparameter Tuning Tests
# ============================================================================

class TestHyperparameterTuning:
    """Test automated hyperparameter optimization."""

    @pytest.fixture
    def clf_data(self):
        """Generate small classification dataset for testing."""
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=0)
        return X, y

    @pytest.fixture
    def reg_data(self):
        """Generate small regression dataset."""
        X, y = make_regression(n_samples=100, n_features=20, random_state=0)
        return X, y

    def test_search_space_classifier(self):
        """Test parameter grid generation for classifiers."""
        for model in ["rf", "svm_rbf", "gboost", "mlp", "knn", "logreg"]:
            space = get_search_space_classifier(model)
            assert isinstance(space, dict)
            assert len(space) > 0

    def test_search_space_regressor(self):
        """Test parameter grid generation for regressors."""
        for model in ["rf_reg", "svr", "mlp_reg", "ridge", "lasso"]:
            space = get_search_space_regressor(model)
            assert isinstance(space, dict)
            assert len(space) > 0

    def test_grid_search_classifier(self, clf_data):
        """Test grid search for classifier."""
        X, y = clf_data
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=0)),
        ])
        
        best_model, results = grid_search_classifier(
            pipeline, X, y, "rf", cv=2, n_jobs=1
        )
        
        assert "best_params" in results
        assert "best_score" in results
        assert 0.0 <= results["best_score"] <= 1.0
        assert hasattr(best_model, "predict")

    def test_quick_tune_classifier(self, clf_data):
        """Test quick tuning with RandomizedSearchCV."""
        X, y = clf_data
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=0)),
        ])
        
        best_model = quick_tune_classifier(pipeline, X, y, "rf", cv=2)
        assert hasattr(best_model, "predict")


# ============================================================================
# GAP 10: Nested Cross-Validation Tests
# ============================================================================

class TestNestedCV:
    """Test nested cross-validation for unbiased model selection."""

    @pytest.fixture
    def clf_data(self):
        """Classification data."""
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=0)
        return X, y

    @pytest.fixture
    def reg_data(self):
        """Regression data."""
        X, y = make_regression(n_samples=100, n_features=20, random_state=0)
        return X, y

    def test_nested_cv_classification(self, clf_data):
        """Test nested CV for classification."""
        X, y = clf_data
        
        estimator = RandomForestClassifier(random_state=0)
        results = nested_cross_validate(estimator, X, y, cv_outer=2, cv_inner=2)
        
        assert "test_scores" in results
        assert "train_scores" in results
        assert "mean_test_score" in results
        assert len(results["test_scores"]) == 2  # 2 outer folds

    def test_nested_cv_regression(self, reg_data):
        """Test nested CV for regression."""
        from sklearn.linear_model import Ridge
        
        X, y = reg_data
        
        estimator = Ridge()
        results = nested_cross_validate_regression(estimator, X, y, cv_outer=2, cv_inner=2)
        
        assert "test_scores" in results
        assert "mean_test_score" in results
        assert len(results["test_scores"]) == 2

    def test_compare_models_nested_cv(self, clf_data):
        """Test comparing multiple models with nested CV."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        
        X, y = clf_data
        
        models = {
            "RF": RandomForestClassifier(random_state=0),
            "Tree": DecisionTreeClassifier(random_state=0),
        }
        
        results = compare_models_nested_cv(
            models, X, y, cv_outer=2, cv_inner=2, task="classification"
        )
        
        assert "RF" in results
        assert "Tree" in results
        assert "mean_test_score" in results["RF"]


# ============================================================================
# GAP 9: Memory Management Tests
# ============================================================================

class TestMemoryManagement:
    """Test memory-efficient hyperspectral processing."""

    @pytest.fixture
    def small_cube(self):
        """Generate small test hyperspectral cube (64x64x10)."""
        return np.random.randn(64, 64, 10).astype(np.float32)

    def test_stream_reader_basic(self, small_cube):
        """Test basic streaming."""
        reader = HyperspectralStreamReader(small_cube, chunk_height=32, chunk_width=32)
        chunks = list(reader.chunks())
        
        assert len(chunks) == 4  # 2x2 chunks
        for chunk, bounds in chunks:
            assert len(chunk.shape) == 3
            assert chunk.shape[2] == 10  # bands preserved

    def test_stream_reader_bounds(self, small_cube):
        """Test that chunk bounds are correct."""
        reader = HyperspectralStreamReader(small_cube, chunk_height=32, chunk_width=32)
        
        bounds_list = [bounds for _, bounds in reader.chunks()]
        # Check coverage
        min_rows = min(b[0] for b in bounds_list)
        max_rows = max(b[1] for b in bounds_list)
        assert min_rows == 0
        assert max_rows == 64

    def test_tiler_basic(self, small_cube):
        """Test tiling without overlap."""
        tiler = HyperspectralTiler(small_cube, tile_height=32, tile_width=32, overlap=0)
        tiles = list(tiler.tiles())
        
        assert len(tiles) == 4
        for tile, bounds in tiles:
            assert tile.shape[2] == 10

    def test_tiler_with_overlap(self, small_cube):
        """Test tiling with overlap."""
        tiler = HyperspectralTiler(small_cube, tile_height=32, tile_width=32, overlap=8)
        tiles = list(tiler.tiles())
        
        assert len(tiles) >= 4
        for tile, bounds in tiles:
            assert tile.shape[2] == 10

    def test_process_chunks(self, small_cube):
        """Test processing chunks."""
        def normalize_chunk(chunk):
            return (chunk - chunk.mean(axis=(0, 1), keepdims=True)) / (
                chunk.std(axis=(0, 1), keepdims=True) + 1e-8
            )
        
        output = process_hyperspectral_chunks(
            small_cube, normalize_chunk, chunk_height=32, chunk_width=32
        )
        
        assert output.shape == small_cube.shape
        # Check normalization worked
        assert np.allclose(output.mean(), 0.0, atol=0.1)

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        memory_mb, human = estimate_memory_usage(512, 512, 100)
        
        assert memory_mb > 0
        assert "MB" in human or "GB" in human
        
        # 512 * 512 * 100 * 4 bytes = ~100 MB
        assert 50 < memory_mb < 150

    def test_chunk_size_recommendation(self):
        """Test chunk size recommendation."""
        chunk_h, chunk_w = recommend_chunk_size(100, available_memory_mb=512)
        
        assert chunk_h > 0
        assert chunk_w > 0
        assert chunk_h == chunk_w  # Should be square


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple gap implementations."""

    def test_threshold_tuning_with_qc_workflow(self):
        """Test threshold tuning in realistic QC workflow."""
        # Simulate health scores from model predictions
        normal_samples = np.random.normal(0.9, 0.05, 50)
        anomalous_samples = np.random.normal(0.2, 0.1, 50)
        
        health_scores = np.concatenate([normal_samples, anomalous_samples])
        labels = np.concatenate([np.ones(50), np.zeros(50)])
        
        # Auto-tune threshold
        threshold = estimate_threshold_youden(health_scores, labels)
        metrics = validate_threshold(health_scores, labels, threshold)
        
        assert metrics["f1_score"] > 0.5  # Should have reasonable F1
        assert 0.2 < threshold < 0.9  # Threshold in reasonable range

    def test_nested_cv_with_tuning(self):
        """Test nested CV with hyperparameter tuning."""
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=0)
        
        # Simple nested CV (no explicit tuning, but demonstrates unbiased evaluation)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=0)),
        ])
        
        # This tests the nested CV workflow
        results = nested_cross_validate(pipeline, X, y, cv_outer=3, cv_inner=2)
        
        # Results should show no overfitting
        assert results["mean_train_score"] > results["mean_test_score"]  # Normal behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
