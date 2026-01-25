"""
Tests for evaluate_model_nested_cv function with nested cross-validation.
Verifies: no leakage, best params recorded, deterministic selection.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.calibration import CalibratedClassifierCV

from foodspec.validation.evaluation import evaluate_model_nested_cv


class TestNestedCVBasic:
    """Test basic nested CV functionality."""

    def test_basic_nested_cv_returns_result(self):
        """Nested CV should return EvaluationResult with hyperparameters."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            seed=42
        )

        # Check result structure
        assert hasattr(result, 'fold_predictions')
        assert hasattr(result, 'fold_metrics')
        assert hasattr(result, 'bootstrap_ci')
        assert hasattr(result, 'hyperparameters_per_fold')
        assert result.hyperparameters_per_fold is not None

    def test_returns_correct_number_of_outer_folds(self):
        """Should return metrics and hyperparameters for each outer fold."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            seed=42
        )

        assert len(result.fold_metrics) == 3
        assert len(result.hyperparameters_per_fold) == 3

    def test_best_params_recorded_per_fold(self):
        """Each outer fold should have recorded hyperparameters."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0, max_iter=100: LogisticRegression(
            C=C, max_iter=max_iter, random_state=42
        )
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0], 'max_iter': [100, 200]},
            seed=42
        )

        for params in result.hyperparameters_per_fold:
            assert 'C' in params
            assert 'max_iter' in params
            assert params['C'] in [0.1, 1.0, 10.0]
            assert params['max_iter'] in [100, 200]

    def test_all_samples_predicted_once(self):
        """Each sample in X should appear exactly once in predictions."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            seed=42
        )

        sample_indices = [pred['sample_idx'] for pred in result.fold_predictions]
        assert len(sample_indices) == 60
        assert len(set(sample_indices)) == 60  # All unique


class TestNestedCVDeterminism:
    """Test deterministic behavior with seed."""

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical results."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result1 = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            seed=42
        )

        result2 = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            seed=42
        )

        # Check that same seed produces same results
        for params1, params2 in zip(result1.hyperparameters_per_fold, result2.hyperparameters_per_fold):
            assert params1 == params2

        # Check predictions are identical
        for pred1, pred2 in zip(result1.fold_predictions, result2.fold_predictions):
            assert pred1['y_pred'] == pred2['y_pred']

    def test_deterministic_best_params_selection(self):
        """Best parameters should be selected deterministically."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            seed=42
        )

        # Best params should be from the grid
        for params in result.hyperparameters_per_fold:
            assert params['C'] in [0.1, 1.0, 10.0]


class TestNestedCVLeakageDetection:
    """Test that inner CV uses only outer training data (no leakage)."""

    def test_inner_cv_uses_outer_train_only(self):
        """Inner CV should never use outer test data (leakage test)."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        # Tracking extractor to verify fit is called correctly
        class TrackingExtractor:
            fit_on_indices = []

            def __init__(self):
                self.mean_ = None

            def fit(self, X, y=None):
                TrackingExtractor.fit_on_indices.append(X.shape[0])
                self.mean_ = np.mean(X, axis=0)
                return self

            def transform(self, X):
                if self.mean_ is None:
                    raise ValueError("Must fit before transform")
                return X - self.mean_

        TrackingExtractor.fit_on_indices = []
        extractor = TrackingExtractor()

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},  # 3 param combinations
            feature_extractor=extractor,
            seed=42
        )

        # Key test: We should have fits recorded, and no data > 40 samples should be fit
        # This ensures inner CV never uses outer test data
        assert len(TrackingExtractor.fit_on_indices) > 0
        
        # All fits should use at most 40 samples (outer train set size)
        # None should use 60 samples (full dataset)
        max_size = max(TrackingExtractor.fit_on_indices)
        assert max_size == 40, f"Max fit size is {max_size}, should be 40 (outer train only)"
        
        # We should have both outer and inner fits
        # Outer fits: 3 (one per outer fold)
        # Inner fits: 3 outer folds * 3 params * 2 inner splits = 18
        # Total: 21
        assert len(TrackingExtractor.fit_on_indices) == 21


class TestNestedCVMetrics:
    """Test metric computation in nested CV."""

    def test_default_metrics_computed(self):
        """Default metrics should include accuracy, macro_f1, auroc_macro."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            seed=42
        )

        for fold_metrics in result.fold_metrics:
            assert 'accuracy' in fold_metrics
            assert 'macro_f1' in fold_metrics
            assert 'auroc_macro' in fold_metrics

    def test_custom_metrics_computed(self):
        """Custom metrics list should be computed."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            metrics=['accuracy', 'precision_macro', 'recall_macro'],
            seed=42
        )

        for fold_metrics in result.fold_metrics:
            assert 'accuracy' in fold_metrics
            assert 'precision_macro' in fold_metrics
            assert 'recall_macro' in fold_metrics

    def test_bootstrap_ci_computed(self):
        """Bootstrap CIs should be computed for each metric."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            seed=42
        )

        assert 'accuracy' in result.bootstrap_ci
        assert 'macro_f1' in result.bootstrap_ci
        assert isinstance(result.bootstrap_ci['accuracy'], tuple)
        assert len(result.bootstrap_ci['accuracy']) == 3  # (lower, median, upper)


class TestNestedCVTuningMetrics:
    """Test different tuning metrics."""

    def test_accuracy_tuning_metric(self):
        """Should support accuracy as tuning metric."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            tuning_metric='accuracy',
            seed=42
        )

        assert len(result.hyperparameters_per_fold) == 3

    def test_macro_f1_tuning_metric(self):
        """Should support macro_f1 as tuning metric."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            tuning_metric='macro_f1',
            seed=42
        )

        assert len(result.hyperparameters_per_fold) == 3


class TestNestedCVGroupHandling:
    """Test group-aware CV for LOBO."""

    def test_groups_tracked_in_predictions(self):
        """Groups should be tracked in per-sample predictions."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)
        groups = np.array(['batch_A'] * 30 + ['batch_B'] * 30)
        meta = pd.DataFrame({'group': groups})

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        # Use GroupKFold for outer (respects groups), StratifiedKFold for inner (ignores groups)
        outer_splitter = GroupKFold(n_splits=2)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0]},
            meta=meta,
            seed=42
        )

        # Check that group is in predictions
        for pred in result.fold_predictions:
            assert 'group' in pred
            assert pred['group'] in ['batch_A', 'batch_B']

    def test_without_groups_no_group_in_predictions(self):
        """Without metadata, group should not be in predictions."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0]},
            seed=42
        )

        for pred in result.fold_predictions:
            assert 'group' not in pred


class TestNestedCVCalibration:
    """Test probability calibration in nested CV."""

    def test_calibrator_applied(self):
        """Calibrator can be used during nested CV (no deepcopy issues)."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        # Use a simple custom calibrator that works with deepcopy
        class SimpleCalibrator:
            def __init__(self):
                self.shift_ = 0.0

            def fit(self, proba, y):
                self.shift_ = 0.0  # No-op calibration for testing
                return self

            def transform(self, proba):
                return proba + self.shift_

        calibrator = SimpleCalibrator()

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0]},
            calibrator=calibrator,
            seed=42
        )

        # Should complete without error and have predictions
        assert len(result.fold_predictions) > 0


class TestNestedCVErrorHandling:
    """Test error handling."""

    def test_invalid_metric_name_raises_error(self):
        """Invalid metric name should raise ValueError."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        with pytest.raises(ValueError):
            evaluate_model_nested_cv(
                X, y, model_factory, outer_splitter, inner_splitter,
                param_grid={'C': [0.1, 1.0]},
                metrics=['invalid_metric'],
                seed=42
            )

    def test_invalid_tuning_metric_raises_error(self):
        """Invalid tuning metric should raise ValueError."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        with pytest.raises(ValueError):
            evaluate_model_nested_cv(
                X, y, model_factory, outer_splitter, inner_splitter,
                param_grid={'C': [0.1, 1.0]},
                tuning_metric='invalid_metric',
                seed=42
            )

    def test_mismatched_lengths_raises_error(self):
        """X and y with different lengths should raise ValueError."""
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 50)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        with pytest.raises(ValueError):
            evaluate_model_nested_cv(
                X, y, model_factory, outer_splitter, inner_splitter,
                param_grid={'C': [0.1, 1.0]},
                seed=42
            )


class TestNestedCVIntegration:
    """Integration tests for nested CV."""

    def test_full_pipeline_with_all_components(self):
        """Nested CV should work with feature extractor and selector."""
        from sklearn.feature_selection import SelectKBest, f_classif

        np.random.seed(42)
        X = np.random.randn(60, 20)
        y = np.random.randint(0, 2, 60)

        class SimpleExtractor:
            def __init__(self):
                self.scale_ = None

            def fit(self, X, y=None):
                self.scale_ = np.std(X, axis=0)
                return self

            def transform(self, X):
                return X / (self.scale_ + 1e-10)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        extractor = SimpleExtractor()
        selector = SelectKBest(f_classif, k=5)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0]},
            feature_extractor=extractor,
            selector=selector,
            seed=42
        )

        # Should have predictions from all outer folds
        assert len(result.fold_predictions) == 60
        assert len(result.fold_metrics) == 3

    def test_realistic_spectroscopy_workflow(self):
        """Test realistic spectroscopy workflow with nested CV."""
        np.random.seed(42)
        n_samples = 100
        n_wavenumbers = 100
        X = np.random.randn(n_samples, n_wavenumbers)
        y = np.random.randint(0, 3, n_samples)  # Multiclass

        model_factory = lambda C=1.0: LogisticRegression(
            C=C, random_state=42, max_iter=200
        )
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            metrics=['accuracy', 'macro_f1'],
            seed=42
        )

        # Check multiclass handling
        assert len(result.fold_metrics) == 3
        for pred in result.fold_predictions:
            assert 'y_true' in pred
            assert 'y_pred' in pred
            assert 0 <= pred['y_pred'] < 3

    def test_grid_search_vs_randomized(self):
        """Test that both grid and randomized search work."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        # Grid search
        result_grid = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={'C': [0.1, 1.0, 10.0]},
            search_strategy='grid',
            seed=42
        )

        # Randomized search
        from scipy.stats import loguniform
        result_random = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_distributions={'C': loguniform(0.01, 100)},
            search_strategy='randomized',
            seed=42
        )

        # Both should return valid results
        assert len(result_grid.fold_metrics) == 3
        assert len(result_random.fold_metrics) == 3

    def test_invalid_search_strategy_raises_error(self):
        """Invalid search_strategy should raise ValueError."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        with pytest.raises(ValueError):
            evaluate_model_nested_cv(
                X, y, model_factory, outer_splitter, inner_splitter,
                param_grid={'C': [0.1, 1.0]},
                search_strategy='invalid',
                seed=42
            )

    def test_empty_param_grid_defaults_to_no_tuning(self):
        """Empty param_grid should use default model factory params."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid={},  # Empty grid
            seed=42
        )

        # Should still work with default params
        assert len(result.fold_metrics) == 3
        assert len(result.hyperparameters_per_fold) == 3
        for params in result.hyperparameters_per_fold:
            assert params == {}  # Default params (no tuning)
