"""
Tests for nested cross-validation runner.
"""

import numpy as np
import pytest
from pathlib import Path

from foodspec.validation.nested import (
    GridSearchTuner,
    NestedCVRunner,
)


class MockEstimator:
    """Mock estimator for testing."""

    def __init__(self, C: float = 1.0, max_iter: int = 100, random_state: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.classes_ = None
        self.coef_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Mock fit."""
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state + int(self.C * 10))
        self.coef_ = rng.standard_normal((len(self.classes_), X.shape[1]))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mock predict_proba with deterministic output based on hyperparameters."""
        rng = np.random.default_rng(self.random_state + int(self.C * 10))
        n_classes = len(self.classes_) if self.classes_ is not None else 2
        proba = rng.dirichlet(np.ones(n_classes), size=X.shape[0])
        # Better performance for C=1.0 (for testing hyperparameter selection)
        if np.isclose(self.C, 1.0):
            proba[:, 0] += 0.1  # Bias toward correct class for testing
            proba /= proba.sum(axis=1, keepdims=True)
        return proba


def mock_estimator_factory(**params):
    """Factory for mock estimator."""
    return MockEstimator(**params)


class TestGridSearchTuner:
    """Test GridSearchTuner hyperparameter selection."""

    def test_select_hyperparameters_returns_best(self):
        """GridSearchTuner should select best hyperparameters from grid."""
        param_grid = {"C": [0.1, 1.0, 10.0], "max_iter": [100]}
        tuner = GridSearchTuner(
            estimator_factory=mock_estimator_factory,
            param_grid=param_grid,
            n_inner_splits=2,
            metric="accuracy",
            seed=42,
        )

        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)

        best_params = tuner.select_hyperparameters(X, y, None)

        assert "C" in best_params
        assert "max_iter" in best_params
        assert best_params["C"] in param_grid["C"]
        assert best_params["max_iter"] == 100

    def test_select_hyperparameters_empty_grid(self):
        """GridSearchTuner with empty grid returns empty dict."""
        tuner = GridSearchTuner(
            estimator_factory=mock_estimator_factory,
            param_grid={},
            n_inner_splits=2,
            seed=42,
        )

        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)

        best_params = tuner.select_hyperparameters(X, y, None)
        assert best_params == {}

    def test_select_hyperparameters_deterministic(self):
        """GridSearchTuner with same seed produces same results."""
        param_grid = {"C": [0.1, 1.0, 10.0]}
        X = np.random.randn(40, 5)
        y = np.random.randint(0, 2, 40)

        tuner1 = GridSearchTuner(
            estimator_factory=mock_estimator_factory,
            param_grid=param_grid,
            n_inner_splits=3,
            seed=123,
        )
        best1 = tuner1.select_hyperparameters(X, y, None)

        tuner2 = GridSearchTuner(
            estimator_factory=mock_estimator_factory,
            param_grid=param_grid,
            n_inner_splits=3,
            seed=123,
        )
        best2 = tuner2.select_hyperparameters(X, y, None)

        assert best1 == best2

    def test_select_hyperparameters_with_groups(self):
        """GridSearchTuner respects group structure."""
        param_grid = {"C": [0.1, 1.0]}
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)
        groups = np.repeat([0, 1, 2, 3, 4], 6)  # 5 groups, 6 samples each

        tuner = GridSearchTuner(
            estimator_factory=mock_estimator_factory,
            param_grid=param_grid,
            n_inner_splits=3,
            seed=42,
        )

        best_params = tuner.select_hyperparameters(X, y, groups)
        assert "C" in best_params


class TestNestedCVRunner:
    """Test NestedCVRunner nested cross-validation."""

    def test_nested_cv_basic_execution(self):
        """NestedCVRunner executes nested CV and returns correct structure."""
        runner = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid={"C": [0.1, 1.0, 10.0]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=42,
        )

        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)

        result = runner.evaluate(X, y)

        # Check structure
        assert len(result.fold_predictions) == 60  # All samples predicted once
        assert len(result.fold_metrics) == 3  # 3 outer folds
        assert len(result.hyperparameters_per_fold) == 3  # Hyperparams for each outer fold
        assert "accuracy" in result.bootstrap_ci

    def test_nested_cv_hyperparameters_recorded(self):
        """NestedCVRunner records hyperparameters for each outer fold."""
        runner = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid={"C": [0.1, 1.0], "max_iter": [100, 200]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=42,
        )

        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)

        result = runner.evaluate(X, y)

        # Each outer fold should have hyperparameters
        for fold_params in result.hyperparameters_per_fold:
            assert "C" in fold_params
            assert "max_iter" in fold_params
            assert fold_params["C"] in [0.1, 1.0]
            assert fold_params["max_iter"] in [100, 200]

    def test_nested_cv_without_tuning(self):
        """NestedCVRunner without param_grid runs standard CV."""
        runner = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid=None,  # No hyperparameter tuning
            n_outer_splits=3,
            seed=42,
        )

        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)

        result = runner.evaluate(X, y)

        # Should still work, but hyperparameters should be empty
        assert len(result.hyperparameters_per_fold) == 3
        for fold_params in result.hyperparameters_per_fold:
            assert fold_params == {}

    def test_nested_cv_deterministic_with_seed(self):
        """NestedCVRunner with same seed produces identical results."""
        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)

        runner1 = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid={"C": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=999,
        )
        result1 = runner1.evaluate(X, y)

        runner2 = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid={"C": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=999,
        )
        result2 = runner2.evaluate(X, y)

        # Same fold metrics
        for m1, m2 in zip(result1.fold_metrics, result2.fold_metrics):
            assert m1["accuracy"] == m2["accuracy"]
            assert m1["macro_f1"] == m2["macro_f1"]

        # Same hyperparameters selected
        assert result1.hyperparameters_per_fold == result2.hyperparameters_per_fold

    def test_nested_cv_with_groups(self):
        """NestedCVRunner respects group structure in both loops."""
        runner = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid={"C": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=42,
        )

        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)
        groups = np.repeat([0, 1, 2, 3, 4, 5], 10)  # 6 groups

        result = runner.evaluate(X, y, groups)

        # Should complete successfully
        assert len(result.fold_metrics) == 3
        assert len(result.hyperparameters_per_fold) == 3

    def test_nested_cv_metrics_on_outer_loop_only(self):
        """NestedCVRunner reports metrics only on outer test sets."""
        runner = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid={"C": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=42,
        )

        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)

        result = runner.evaluate(X, y)

        # Each outer fold should have exactly one metric entry
        fold_ids = [m["fold_id"] for m in result.fold_metrics]
        assert fold_ids == [0, 1, 2]

        # All predictions should be from outer test sets
        assert len(result.fold_predictions) == 60

    def test_nested_cv_saves_artifacts(self, tmp_path):
        """NestedCVRunner saves artifacts including hyperparameters."""
        output_dir = tmp_path / "nested_cv_run"

        runner = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid={"C": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=42,
            output_dir=output_dir,
        )

        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)

        result = runner.evaluate(X, y)

        # Check artifacts saved
        assert output_dir.exists()
        assert (output_dir / "predictions.csv").exists()
        assert (output_dir / "metrics.csv").exists()
        assert (output_dir / "hyperparameters_per_fold.csv").exists()

        # Load and verify hyperparameters
        import pandas as pd

        hyperparams_df = pd.read_csv(output_dir / "hyperparameters_per_fold.csv")
        assert len(hyperparams_df) == 3
        assert "fold_id" in hyperparams_df.columns
        assert "C" in hyperparams_df.columns

    def test_nested_cv_input_validation(self):
        """NestedCVRunner validates input shapes."""
        runner = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            n_outer_splits=3,
            seed=42,
        )

        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 40)  # Wrong length

        with pytest.raises(ValueError, match="X and y must have the same length"):
            runner.evaluate(X, y)

    def test_nested_cv_bootstrap_ci_computed(self):
        """NestedCVRunner computes bootstrap CIs."""
        runner = NestedCVRunner(
            estimator_factory=mock_estimator_factory,
            param_grid={"C": [0.1, 1.0]},
            n_outer_splits=4,
            seed=42,
        )

        X = np.random.randn(80, 5)
        y = np.random.randint(0, 2, 80)

        result = runner.evaluate(X, y)

        # Bootstrap CIs should be present
        assert "accuracy" in result.bootstrap_ci
        assert "macro_f1" in result.bootstrap_ci

        # CI should be a tuple (lower, upper)
        acc_ci = result.bootstrap_ci["accuracy"]
        assert len(acc_ci) == 2
        assert acc_ci[0] < acc_ci[1]  # lower < upper

    def test_nested_cv_different_tuning_metrics(self):
        """NestedCVRunner supports different tuning metrics."""
        for metric in ["accuracy", "macro_f1", "auroc"]:
            runner = NestedCVRunner(
                estimator_factory=mock_estimator_factory,
                param_grid={"C": [0.1, 1.0]},
                n_outer_splits=3,
                n_inner_splits=2,
                tuning_metric=metric,
                seed=42,
            )

            X = np.random.randn(60, 5)
            y = np.random.randint(0, 2, 60)

            result = runner.evaluate(X, y)

            # Should complete successfully
            assert len(result.hyperparameters_per_fold) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
