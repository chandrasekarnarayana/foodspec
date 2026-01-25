"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

Nested cross-validation for unbiased performance estimation with hyperparameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.validation.metrics import accuracy, auroc_macro, macro_f1
from foodspec.validation.splits import StratifiedKFoldOrGroupKFold


class Estimator(Protocol):
    """Protocol for a scikit-learn-like estimator."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class HyperparameterTuner(Protocol):
    """Protocol for hyperparameter tuning strategy."""

    def select_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, groups_train: Sequence[object] | None
    ) -> Dict[str, Any]:
        """Select best hyperparameters via inner CV.

        Parameters
        ----------
        X_train : ndarray
            Training features for inner CV.
        y_train : ndarray
            Training labels for inner CV.
        groups_train : sequence or None
            Group labels for inner CV.

        Returns
        -------
        best_params : dict
            Selected hyperparameters.
        """
        ...


class GridSearchTuner:
    """Simple grid search hyperparameter tuner.

    Parameters
    ----------
    estimator_factory : callable
        Function that takes hyperparameters as kwargs and returns an Estimator.
    param_grid : dict
        Dictionary of hyperparameter names to lists of values to try.
    n_inner_splits : int, default 3
        Number of CV splits for inner loop.
    metric : str, default "accuracy"
        Metric to optimize ('accuracy', 'macro_f1', 'auroc').
    seed : int, default 0
        Random seed for reproducibility.

    Examples
    --------
    >>> from foodspec.models import LogisticRegressionClassifier
    >>> tuner = GridSearchTuner(
    ...     estimator_factory=lambda **params: LogisticRegressionClassifier(**params),
    ...     param_grid={"C": [0.1, 1.0, 10.0], "max_iter": [100, 200]},
    ...     n_inner_splits=3,
    ... )
    >>> X = np.random.randn(50, 5)
    >>> y = np.random.randint(0, 2, 50)
    >>> best = tuner.select_hyperparameters(X, y, groups=None)
    >>> "C" in best
    True
    """

    def __init__(
        self,
        estimator_factory: Any,
        param_grid: Dict[str, List[Any]],
        n_inner_splits: int = 3,
        metric: str = "accuracy",
        seed: int = 0,
    ):
        self.estimator_factory = estimator_factory
        self.param_grid = param_grid
        self.n_inner_splits = n_inner_splits
        self.metric = metric
        self.seed = seed

    def select_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, groups_train: Sequence[object] | None
    ) -> Dict[str, Any]:
        """Select best hyperparameters via inner CV grid search."""

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(self.param_grid)

        if not param_combinations:
            return {}  # No hyperparameters to tune

        # Evaluate each combination via inner CV
        best_score = -np.inf
        best_params = param_combinations[0]

        for params in param_combinations:
            scores = self._evaluate_params(params, X_train, y_train, groups_train)
            avg_score = np.mean(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        return best_params

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations from parameter grid."""
        if not param_grid:
            return [{}]

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        combinations = []
        self._recursive_combinations(keys, values, 0, {}, combinations)
        return combinations

    def _recursive_combinations(
        self,
        keys: List[str],
        values: List[List[Any]],
        idx: int,
        current: Dict[str, Any],
        result: List[Dict[str, Any]],
    ) -> None:
        """Recursively generate all combinations."""
        if idx == len(keys):
            result.append(current.copy())
            return

        for val in values[idx]:
            current[keys[idx]] = val
            self._recursive_combinations(keys, values, idx + 1, current, result)
            del current[keys[idx]]

    def _evaluate_params(
        self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, groups: Sequence[object] | None
    ) -> List[float]:
        """Evaluate hyperparameters via inner CV."""

        splitter = StratifiedKFoldOrGroupKFold(n_splits=self.n_inner_splits, seed=self.seed)
        scores = []

        for train_idx, val_idx in splitter.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train with these hyperparameters
            estimator = self.estimator_factory(**params)
            estimator.fit(X_train, y_train)
            proba = estimator.predict_proba(X_val)
            pred = proba.argmax(axis=1) if proba.ndim == 2 else (proba > 0.5).astype(int)

            # Compute metric
            if self.metric == "accuracy":
                score_dict = accuracy(y_val, pred, proba)
                score = score_dict["accuracy"]
            elif self.metric == "macro_f1":
                score_dict = macro_f1(y_val, pred, proba)
                score = score_dict["macro_f1"]
            elif self.metric == "auroc":
                try:
                    score_dict = auroc_macro(y_val, pred, proba)
                    score = score_dict["auroc_macro"]
                except Exception:
                    score = 0.0
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            scores.append(score)

        return scores


@dataclass
class NestedCVResult:
    """Result of nested cross-validation.

    Attributes
    ----------
    fold_predictions : list of dict
        Per-sample predictions from outer loop test sets.
    fold_metrics : list of dict
        Per-fold metrics computed on outer test sets.
    hyperparameters_per_fold : list of dict
        Hyperparameters chosen in inner CV for each outer fold.
    bootstrap_ci : dict
        Bootstrap confidence intervals for aggregate metrics.
    """

    fold_predictions: List[Dict[str, Any]]
    fold_metrics: List[Dict[str, float]]
    hyperparameters_per_fold: List[Dict[str, Any]]
    bootstrap_ci: Dict[str, Tuple[float, float]]


@dataclass
class NestedCVRunner:
    """Nested cross-validation runner for unbiased hyperparameter tuning.

    Nested CV consists of:
    - **Outer loop**: Splits data for unbiased performance estimation
    - **Inner loop**: For each outer fold, run inner CV to select hyperparameters
    - Final model for each outer fold is trained on full outer training set with best params
    - Metrics reported only on outer test sets (unbiased)

    Parameters
    ----------
    estimator_factory : callable
        Function that takes hyperparameters as kwargs and returns an Estimator.
    param_grid : dict, optional
        Hyperparameter grid for tuning. If None or empty, no tuning (standard CV).
    n_outer_splits : int, default 5
        Number of outer CV folds.
    n_inner_splits : int, default 3
        Number of inner CV folds for hyperparameter tuning.
    tuning_metric : str, default "accuracy"
        Metric to optimize in inner CV ('accuracy', 'macro_f1', 'auroc').
    seed : int, default 0
        Random seed for deterministic splits.
    output_dir : Path | str | None, default None
        Directory for artifact output. If None, no artifacts are saved.

    Examples
    --------
    >>> from foodspec.models import LogisticRegressionClassifier
    >>> import numpy as np
    >>> runner = NestedCVRunner(
    ...     estimator_factory=lambda **p: LogisticRegressionClassifier(**p),
    ...     param_grid={"C": [0.1, 1.0, 10.0]},
    ...     n_outer_splits=3,
    ...     n_inner_splits=2,
    ...     seed=42,
    ... )
    >>> X = np.random.randn(60, 5)
    >>> y = np.random.randint(0, 2, 60)
    >>> result = runner.evaluate(X, y)
    >>> len(result.hyperparameters_per_fold)
    3
    >>> "C" in result.hyperparameters_per_fold[0]
    True
    """

    estimator_factory: Any  # Callable that returns Estimator
    param_grid: Dict[str, List[Any]] | None = None
    n_outer_splits: int = 5
    n_inner_splits: int = 3
    tuning_metric: str = "accuracy"
    seed: int = 0
    output_dir: Path | str | None = None
    stability_selector: Optional[Any] = None
    x_wavenumbers: Sequence[float] | None = None

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, groups: Sequence[object] | None = None
    ) -> NestedCVResult:
        """Run nested cross-validation.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : ndarray, shape (n_samples,)
            Labels.
        groups : sequence, optional
            Group labels for grouped CV splits.

        Returns
        -------
        result : NestedCVResult
            Predictions, metrics, and hyperparameters from nested CV.

        Raises
        ------
        ValueError
            If X and y have incompatible shapes or if metric is unknown.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same length; got {X.shape[0]} and {y.shape[0]}")

        # Set up artifacts if output directory provided
        if self.output_dir:
            artifact_dir = Path(self.output_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifacts = ArtifactRegistry(artifact_dir)
            artifacts.ensure_layout()
        else:
            artifacts = None

        # Set up hyperparameter tuner if param_grid provided
        if self.param_grid:
            tuner: HyperparameterTuner = GridSearchTuner(
                estimator_factory=self.estimator_factory,
                param_grid=self.param_grid,
                n_inner_splits=self.n_inner_splits,
                metric=self.tuning_metric,
                seed=self.seed,
            )
        else:
            tuner = None

        # Outer CV loop
        outer_splitter = StratifiedKFoldOrGroupKFold(n_splits=self.n_outer_splits, seed=self.seed)
        fold_predictions: List[Dict[str, Any]] = []
        fold_metrics: List[Dict[str, float]] = []
        hyperparameters_per_fold: List[Dict[str, Any]] = []
        accuracies, macro_f1s, aurocs = [], [], []

        for outer_fold_id, (train_idx, test_idx) in enumerate(outer_splitter.split(X, y, groups)):
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            groups_train_outer = [groups[i] for i in train_idx] if groups is not None else None

            # Inner CV loop: Hyperparameter selection
            if tuner:
                best_params = tuner.select_hyperparameters(
                    X_train_outer, y_train_outer, groups_train_outer
                )
            else:
                best_params = {}  # No tuning

            hyperparameters_per_fold.append(best_params)

            # Train final model on full outer training set with best hyperparameters
            estimator = self.estimator_factory(**best_params)
            estimator.fit(X_train_outer, y_train_outer)

            # Predict on outer test set (unbiased evaluation)
            proba = estimator.predict_proba(X_test_outer)
            pred = proba.argmax(axis=1) if proba.ndim == 2 else (proba > 0.5).astype(int)

            # Compute metrics on outer test set
            acc_dict = accuracy(y_test_outer, pred, proba)
            f1_dict = macro_f1(y_test_outer, pred, proba)
            try:
                roc_dict = auroc_macro(y_test_outer, pred, proba)
                roc = roc_dict["auroc_macro"]
            except Exception:
                roc = np.nan

            acc = acc_dict["accuracy"]
            f1 = f1_dict["macro_f1"]

            accuracies.append(acc)
            macro_f1s.append(f1)
            aurocs.append(roc)

            # Store per-fold metrics
            metrics_row = {
                "fold_id": outer_fold_id,
                "accuracy": acc,
                "macro_f1": f1,
                "auroc": roc,
            }
            fold_metrics.append(metrics_row)

            # Store per-sample predictions
            for i, idx in enumerate(test_idx):
                pred_row = {
                    "fold_id": outer_fold_id,
                    "sample_idx": int(idx),
                    "y_true": int(y_test_outer[i]),
                    "y_pred": int(pred[i]),
                    "proba_0": float(proba[i, 0]) if proba.ndim == 2 else (1.0 - float(proba[i])),
                }
                if proba.ndim == 2 and proba.shape[1] > 1:
                    pred_row["proba_1"] = float(proba[i, 1])
                fold_predictions.append(pred_row)

        # Bootstrap CIs for aggregate metrics
        bootstrap_ci_vals = {}
        valid_f1s = np.array([f for f in macro_f1s if not np.isnan(f)])
        valid_aurocs = np.array([r for r in aurocs if not np.isnan(r)])
        if len(valid_f1s) > 0:
            bootstrap_ci_vals["macro_f1"] = self._bootstrap_ci(valid_f1s)
        if len(valid_aurocs) > 0:
            bootstrap_ci_vals["auroc"] = self._bootstrap_ci(valid_aurocs)
        bootstrap_ci_vals["accuracy"] = self._bootstrap_ci(np.array(accuracies))

        # Save artifacts
        if artifacts:
            artifacts.write_csv(artifacts.predictions_path, fold_predictions)
            artifacts.write_csv(artifacts.metrics_path, fold_metrics)

            # Save hyperparameters per fold
            hyperparams_path = Path(self.output_dir) / "hyperparameters_per_fold.csv"
            hyperparams_records = []
            for fold_id, params in enumerate(hyperparameters_per_fold):
                record = {"fold_id": fold_id, **params}
                hyperparams_records.append(record)
            artifacts.write_csv(hyperparams_path, hyperparams_records)

            # Optional: run stability selection across outer folds and save marker_panel.json
            if self.stability_selector is not None:
                # Local import to avoid circular dependency
                from foodspec.features.selection import run_stability_selection_cv
                
                run_stability_selection_cv(
                    estimator_factory=self.stability_selector.estimator_factory,
                    selector=self.stability_selector,
                    X=X,
                    y=y,
                    x_wavenumbers=self.x_wavenumbers,
                    n_splits=self.n_outer_splits,
                    seed=self.seed,
                    groups=groups,
                    output_dir=self.output_dir,
                )

        return NestedCVResult(
            fold_predictions=fold_predictions,
            fold_metrics=fold_metrics,
            hyperparameters_per_fold=hyperparameters_per_fold,
            bootstrap_ci=bootstrap_ci_vals,
        )

    def _bootstrap_ci(
        self, values: np.ndarray, n_bootstraps: int = 1000, ci: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        rng = np.random.default_rng(self.seed)
        bootstraps = []
        n = len(values)
        alpha = 1.0 - ci
        for _ in range(n_bootstraps):
            sample = rng.choice(values, size=n, replace=True)
            bootstraps.append(sample.mean())
        bootstraps = np.array(bootstraps)
        lower = float(np.quantile(bootstraps, alpha / 2.0))
        upper = float(np.quantile(bootstraps, 1.0 - alpha / 2.0))
        return (lower, upper)


__all__ = [
    "Estimator",
    "HyperparameterTuner",
    "GridSearchTuner",
    "NestedCVResult",
    "NestedCVRunner",
]
