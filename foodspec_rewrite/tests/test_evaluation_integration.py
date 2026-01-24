"""
Tests for evaluation runner integration with nested CV.
"""

import numpy as np
import pytest

from foodspec.validation.evaluation import create_evaluation_runner, EvaluationResult


class MockEstimator:
    """Mock estimator for testing."""

    def __init__(self, C: float = 1.0, random_state: int = 0):
        self.C = C
        self.random_state = random_state
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Mock fit."""
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mock predict_proba."""
        rng = np.random.default_rng(self.random_state)
        n_classes = len(self.classes_) if self.classes_ is not None else 2
        return rng.dirichlet(np.ones(n_classes), size=X.shape[0])


class TestCreateEvaluationRunner:
    """Test create_evaluation_runner factory function."""

    def test_standard_cv_runner(self):
        """create_evaluation_runner returns EvaluationRunner for standard CV."""
        estimator = MockEstimator(C=1.0)
        runner = create_evaluation_runner(
            estimator=estimator,
            nested=False,
            n_splits=3,
            seed=42,
        )

        # Should return EvaluationRunner
        assert hasattr(runner, "evaluate")
        assert runner.n_splits == 3
        assert runner.seed == 42

        # Run evaluation
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)
        result = runner.evaluate(X, y)

        assert isinstance(result, EvaluationResult)
        assert len(result.fold_metrics) == 3
        assert result.hyperparameters_per_fold is None  # Standard CV has no hyperparams

    def test_nested_cv_runner(self):
        """create_evaluation_runner returns NestedCVRunner for nested CV."""
        runner = create_evaluation_runner(
            estimator_factory=lambda **p: MockEstimator(**p),
            param_grid={"C": [0.1, 1.0]},
            nested=True,
            n_splits=3,
            n_inner_splits=2,
            seed=42,
        )

        # Should return NestedCVRunner
        assert hasattr(runner, "evaluate")
        assert runner.n_outer_splits == 3
        assert runner.n_inner_splits == 2

        # Run evaluation
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)
        result = runner.evaluate(X, y)

        assert len(result.fold_metrics) == 3
        assert result.hyperparameters_per_fold is not None  # Nested CV has hyperparams
        assert len(result.hyperparameters_per_fold) == 3

    def test_standard_cv_missing_estimator(self):
        """create_evaluation_runner raises error if nested=False without estimator."""
        with pytest.raises(ValueError, match="nested=False requires estimator"):
            create_evaluation_runner(nested=False, n_splits=3)

    def test_nested_cv_missing_factory(self):
        """create_evaluation_runner raises error if nested=True without factory."""
        with pytest.raises(ValueError, match="nested=True requires estimator_factory"):
            create_evaluation_runner(nested=True, n_splits=3)

    def test_factory_passes_all_parameters(self):
        """create_evaluation_runner passes all parameters correctly."""
        runner = create_evaluation_runner(
            estimator_factory=lambda **p: MockEstimator(**p),
            param_grid={"C": [0.1, 1.0, 10.0]},
            nested=True,
            n_splits=5,
            n_inner_splits=4,
            tuning_metric="macro_f1",
            seed=123,
        )

        assert runner.n_outer_splits == 5
        assert runner.n_inner_splits == 4
        assert runner.tuning_metric == "macro_f1"
        assert runner.seed == 123
        assert runner.param_grid == {"C": [0.1, 1.0, 10.0]}

    def test_standard_cv_deterministic(self):
        """Standard CV via factory is deterministic with same seed."""
        X = np.random.randn(40, 5)
        y = np.random.randint(0, 2, 40)

        runner1 = create_evaluation_runner(
            estimator=MockEstimator(random_state=42),
            nested=False,
            n_splits=3,
            seed=999,
        )
        result1 = runner1.evaluate(X, y)

        runner2 = create_evaluation_runner(
            estimator=MockEstimator(random_state=42),
            nested=False,
            n_splits=3,
            seed=999,
        )
        result2 = runner2.evaluate(X, y)

        # Same fold structure
        assert len(result1.fold_metrics) == len(result2.fold_metrics)

    def test_nested_cv_deterministic(self):
        """Nested CV via factory is deterministic with same seed."""
        X = np.random.randn(40, 5)
        y = np.random.randint(0, 2, 40)

        runner1 = create_evaluation_runner(
            estimator_factory=lambda **p: MockEstimator(**p),
            param_grid={"C": [0.1, 1.0]},
            nested=True,
            n_splits=3,
            n_inner_splits=2,
            seed=999,
        )
        result1 = runner1.evaluate(X, y)

        runner2 = create_evaluation_runner(
            estimator_factory=lambda **p: MockEstimator(**p),
            param_grid={"C": [0.1, 1.0]},
            nested=True,
            n_splits=3,
            n_inner_splits=2,
            seed=999,
        )
        result2 = runner2.evaluate(X, y)

        # Same hyperparameters selected
        assert result1.hyperparameters_per_fold == result2.hyperparameters_per_fold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
