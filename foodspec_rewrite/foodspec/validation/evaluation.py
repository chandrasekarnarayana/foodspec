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

Cross-validation evaluation runner with bootstrap confidence intervals and artifact saving.
Supports both standard CV and nested CV with hyperparameter tuning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.features.base import FeatureSet
from foodspec.features.marker_panel import MarkerPanel
from foodspec.validation.metrics import accuracy, auroc_ovr, macro_f1
from foodspec.validation.splits import StratifiedKFoldOrGroupKFold


class Estimator(Protocol):
    """Protocol for a scikit-learn-like estimator."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class EvaluationResult:
    """Holds per-fold predictions and metrics, plus bootstrap CIs.
    
    Attributes
    ----------
    fold_predictions : list of dict
        Per-sample predictions from test sets.
    fold_metrics : list of dict
        Per-fold metrics.
    bootstrap_ci : dict
        Bootstrap confidence intervals.
    hyperparameters_per_fold : list of dict, optional
        Hyperparameters selected per fold (only for nested CV).
    """

    fold_predictions: List[Dict[str, Any]]  # Per-fold pred dicts: idx, y_true, y_pred, proba...
    fold_metrics: List[Dict[str, float]]  # Per-fold metrics: fold_id, accuracy, macro_f1, auroc
    bootstrap_ci: Dict[str, Tuple[float, float]]  # metric -> (lower, upper) CI bounds
    hyperparameters_per_fold: Optional[List[Dict[str, Any]]] = None  # For nested CV


def bootstrap_ci(values: np.ndarray, n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Compute bootstrap CI for a metric (e.g., mean F1 across folds).

    Parameters
    ----------
    values : ndarray
        Metric values across folds.
    n_bootstraps : int, default 1000
        Number of bootstrap resamples.
    ci : float, default 0.95
        Confidence level (e.g., 0.95 for 95% CI).
    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    (lower, upper) : tuple of float
        Confidence interval bounds.
    """

    rng = np.random.default_rng(seed)
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


@dataclass
class EvaluationRunner:
    """Evaluate a model using cross-validation with bootstrap CIs.

    Parameters
    ----------
    estimator : Estimator
        A fitted-capable model with predict_proba.
    n_splits : int, default 5
        Number of CV folds.
    seed : int, default 0
        Random seed for deterministic splits.
    output_dir : Path | str | None, default None
        Directory for artifact output. If None, no artifacts are saved.

    Examples
    --------
    >>> from foodspec.models import LogisticRegressionClassifier
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> clf = LogisticRegressionClassifier(random_state=0)
    >>> runner = EvaluationRunner(clf, n_splits=5, seed=42)
    >>> result = runner.evaluate(X, y, groups=None)
    >>> len(result.fold_metrics)
    5
    >>> "macro_f1" in result.bootstrap_ci
    True
    """

    estimator: Estimator
    n_splits: int = 5
    seed: int = 0
    output_dir: Path | str | None = None
    stability_selector: Optional[Any] = None
    x_wavenumbers: Sequence[float] | None = None
    preprocessors: Optional[Sequence[Any]] = None
    feature_extractors: Optional[Sequence[Any] | Any] = None

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, groups: Sequence[object] | None = None
    ) -> EvaluationResult:
        """Run cross-validation evaluation.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : ndarray, shape (n_samples,)
            Labels.
        groups : sequence, optional
            Group labels for grouped CV. If provided and sufficient unique groups,
            GroupKFold is used; otherwise StratifiedKFold.

        Returns
        -------
        result : EvaluationResult
            Per-fold predictions and metrics, plus bootstrap CIs.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")

        if self.output_dir:
            artifact_dir = Path(self.output_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifacts = ArtifactRegistry(artifact_dir)
            artifacts.ensure_layout()
        else:
            artifacts = None

        splitter = StratifiedKFoldOrGroupKFold(n_splits=self.n_splits, seed=self.seed)
        fold_predictions: List[Dict[str, Any]] = []
        fold_metrics: List[Dict[str, float]] = []
        accuracies, macro_f1s, aurocs = [], [], []

        preprocess_steps = list(self.preprocessors or [])
        feature_extractors = self._normalize_feature_extractors()
        aggregate_selection_freq: Optional[np.ndarray] = None
        base_feature_names: Optional[List[str]] = None

        def _clone(component: Any) -> Any:
            return copy.deepcopy(component)

        def _apply_preprocessors(
            steps: Sequence[Any], X_train_fold: np.ndarray, X_val_fold: np.ndarray, y_train_fold: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            Xt_train, Xt_val = X_train_fold, X_val_fold
            for step in steps:
                transformer = _clone(step)
                needs_fit = hasattr(transformer, "fit") and not getattr(transformer, "stateless", False)
                if needs_fit:
                    try:
                        transformer.fit(Xt_train, y_train_fold)
                    except TypeError:
                        transformer.fit(Xt_train)

                if hasattr(transformer, "transform"):
                    Xt_train = transformer.transform(Xt_train)
                    Xt_val = transformer.transform(Xt_val)
                elif hasattr(transformer, "fit_transform"):
                    Xt_train = transformer.fit_transform(Xt_train, y_train_fold)
                    Xt_val = transformer.transform(Xt_val)
                else:
                    raise TypeError("Preprocessor must implement transform or fit_transform")
            return Xt_train, Xt_val

        def _normalize_feature_output(result: Any) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
            if isinstance(result, FeatureSet):
                return result.Xf, list(result.feature_names), dict(result.feature_meta)
            if isinstance(result, pd.DataFrame):
                return result.values, list(result.columns), {}
            if isinstance(result, tuple) and len(result) == 2:
                Xf, names = result
                return np.asarray(Xf, dtype=float), list(names), {}
            raise TypeError("Feature extractor must return FeatureSet, DataFrame, or (Xf, feature_names) tuple")

        def _run_feature_extractors(
            extractors: Sequence[Any], X_train_fold: np.ndarray, X_val_fold: np.ndarray, y_train_fold: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
            if not extractors:
                default_names = [f"f{i}" for i in range(X_train_fold.shape[1])]
                return X_train_fold, X_val_fold, default_names, {}

            train_feats: List[np.ndarray] = []
            val_feats: List[np.ndarray] = []
            names: List[str] = []
            meta: Dict[str, Any] = {}

            for extractor in extractors:
                ext_instance = _clone(extractor)
                try:
                    ext_instance.fit(X_train_fold, y_train_fold)
                except TypeError:
                    ext_instance.fit(X_train_fold)

                train_result = ext_instance.transform(X_train_fold)
                val_result = ext_instance.transform(X_val_fold)
                X_train_feat, names_train, meta_train = _normalize_feature_output(train_result)
                X_val_feat, names_val, _ = _normalize_feature_output(val_result)
                if names_train != names_val:
                    raise ValueError("Feature names differ between train and validation transforms")

                train_feats.append(X_train_feat)
                val_feats.append(X_val_feat)
                names.extend(names_train)
                meta.update(meta_train)

            return np.hstack(train_feats), np.hstack(val_feats), names, meta

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_proc, X_test_proc = _apply_preprocessors(preprocess_steps, X_train, X_test, y_train)
            X_train_feats, X_test_feats, feature_names, feature_meta = _run_feature_extractors(
                feature_extractors, X_train_proc, X_test_proc, y_train
            )

            selected_feature_names = feature_names

            if self.stability_selector is not None:
                selector = _clone(self.stability_selector)
                selector.fit(X_train_feats, y_train)

                if selector.selection_frequencies_ is not None:
                    if aggregate_selection_freq is None:
                        aggregate_selection_freq = np.zeros_like(selector.selection_frequencies_, dtype=float)
                        base_feature_names = feature_names
                    aggregate_selection_freq += selector.selection_frequencies_

                X_train_feats = selector.transform(X_train_feats)
                X_test_feats = selector.transform(X_test_feats)
                if selector.selected_indices_:
                    selected_feature_names = [feature_names[i] for i in selector.selected_indices_]

                if artifacts:
                    panel_payload = selector.get_marker_panel(
                        x_wavenumbers=self.x_wavenumbers,
                        feature_names=feature_names,
                    )
                    fold_panel_path = artifacts.root / f"marker_panel_fold_{fold_id}.json"
                    artifacts.write_json(fold_panel_path, panel_payload)

            estimator = _clone(self.estimator)
            estimator.fit(X_train_feats, y_train)
            proba = estimator.predict_proba(X_test_feats)
            pred = proba.argmax(axis=1) if proba.ndim == 2 else (proba > 0.5).astype(int)

            acc = accuracy(y_test, proba)
            f1 = macro_f1(y_test, proba)
            try:
                roc = auroc_ovr(y_test, proba)
            except Exception:
                roc = np.nan

            accuracies.append(acc)
            macro_f1s.append(f1)
            aurocs.append(roc)

            metrics_row = {"fold_id": fold_id, "accuracy": acc, "macro_f1": f1, "auroc": roc}
            fold_metrics.append(metrics_row)

            for i, idx in enumerate(test_idx):
                pred_row = {
                    "fold_id": fold_id,
                    "sample_idx": int(idx),
                    "y_true": int(y_test[i]),
                    "y_pred": int(pred[i]),
                    "proba_0": float(proba[i, 0]) if proba.ndim == 2 else (1.0 - float(proba[i])),
                }
                if proba.ndim == 2 and proba.shape[1] > 1:
                    pred_row["proba_1"] = float(proba[i, 1])
                fold_predictions.append(pred_row)

        bootstrap_ci_vals = {}
        valid_f1s = np.array([f for f in macro_f1s if not np.isnan(f)])
        valid_aurocs = np.array([r for r in aurocs if not np.isnan(r)])
        if len(valid_f1s) > 0:
            bootstrap_ci_vals["macro_f1"] = bootstrap_ci(valid_f1s, seed=self.seed)
        if len(valid_aurocs) > 0:
            bootstrap_ci_vals["auroc"] = bootstrap_ci(valid_aurocs, seed=self.seed)
        bootstrap_ci_vals["accuracy"] = bootstrap_ci(np.array(accuracies), seed=self.seed)

        if artifacts:
            artifacts.write_csv(artifacts.predictions_path, fold_predictions)
            artifacts.write_csv(artifacts.metrics_path, fold_metrics)

            if self.stability_selector is not None and aggregate_selection_freq is not None:
                avg_freqs = aggregate_selection_freq / float(self.n_splits)
                feature_names_for_panel = base_feature_names or [f"f{i}" for i in range(len(avg_freqs))]
                selected_indices = [
                    int(i) for i, freq in enumerate(avg_freqs) if freq >= self.stability_selector.selection_threshold
                ]
                selected_feature_names = [feature_names_for_panel[i] for i in selected_indices]

                selected_wavenumbers = None
                if self.x_wavenumbers is not None:
                    x_arr = np.asarray(self.x_wavenumbers, dtype=float)
                    if len(x_arr) >= len(avg_freqs):
                        selected_wavenumbers = [float(x_arr[i]) for i in selected_indices]

                panel = MarkerPanel(
                    selected_feature_names=selected_feature_names,
                    selected_indices=selected_indices,
                    selection_frequencies=avg_freqs.tolist(),
                    selected_wavenumbers=selected_wavenumbers,
                    n_splits=self.n_splits,
                    n_resamples=self.stability_selector.n_resamples,
                    subsample_fraction=self.stability_selector.subsample_fraction,
                    selection_threshold=self.stability_selector.selection_threshold,
                    seed=self.stability_selector.random_state,
                    created_by=self.stability_selector.__class__.__name__,
                    protocol_hash="",
                    extra={"aggregation": "mean_frequency_across_folds"},
                )
                panel.save(artifacts)

        return EvaluationResult(
            fold_predictions=fold_predictions,
            fold_metrics=fold_metrics,
            bootstrap_ci=bootstrap_ci_vals,
            hyperparameters_per_fold=None,
        )

    def _normalize_feature_extractors(self) -> List[Any]:
        """Normalize feature_extractors to a list for iteration."""

        if self.feature_extractors is None:
            return []
        if isinstance(self.feature_extractors, Sequence) and not isinstance(
            self.feature_extractors, (str, bytes)
        ):
            return list(self.feature_extractors)
        return [self.feature_extractors]


def create_evaluation_runner(
    estimator: Optional[Estimator] = None,
    estimator_factory: Optional[Callable[..., Estimator]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    nested: bool = False,
    n_splits: int = 5,
    n_inner_splits: int = 3,
    tuning_metric: str = "accuracy",
    seed: int = 0,
    output_dir: Optional[Path | str] = None,
    stability_selector: Optional[Any] = None,
    x_wavenumbers: Optional[Sequence[float]] = None,
    preprocessors: Optional[Sequence[Any]] = None,
    feature_extractors: Optional[Sequence[Any] | Any] = None,
) -> EvaluationRunner | Any:
    """Create appropriate evaluation runner (standard or nested CV).

    Parameters
    ----------
    estimator : Estimator, optional
        Pre-configured estimator for standard CV.
    estimator_factory : callable, optional
        Factory function for nested CV that takes hyperparameters as kwargs.
    param_grid : dict, optional
        Hyperparameter grid for nested CV.
    nested : bool, default False
        If True, use nested CV with hyperparameter tuning.
    n_splits : int, default 5
        Number of outer CV folds.
    n_inner_splits : int, default 3
        Number of inner CV folds (nested CV only).
    tuning_metric : str, default "accuracy"
        Metric to optimize in inner loop (nested CV only).
    seed : int, default 0
        Random seed for reproducibility.
    output_dir : Path | str | None, default None
        Directory for artifact output.
    preprocessors : sequence, optional
        Preprocessing transformers to fit on training folds and apply to both train/val.
    feature_extractors : sequence or FeatureExtractor, optional
        Feature extractor instances to fit per fold and transform train/val without leakage.

    Returns
    -------
    runner : EvaluationRunner or NestedCVRunner
        Configured evaluation runner.

    Raises
    ------
    ValueError
        If nested=True but estimator_factory not provided, or if
        nested=False but estimator not provided.

    Examples
    --------
    Standard CV::

        from foodspec.models import LogisticRegressionClassifier
        runner = create_evaluation_runner(
            estimator=LogisticRegressionClassifier(C=1.0),
            nested=False,
            n_splits=5,
            seed=42,
        )

    Nested CV with hyperparameter tuning::

        runner = create_evaluation_runner(
            estimator_factory=lambda **p: LogisticRegressionClassifier(**p),
            param_grid={"C": [0.1, 1.0, 10.0]},
            nested=True,
            n_splits=5,
            n_inner_splits=3,
            seed=42,
        )
    """

    if nested:
        # Use nested CV
        if estimator_factory is None:
            raise ValueError(
                "nested=True requires estimator_factory (callable that takes hyperparameters as kwargs)"
            )

        # Import here to avoid circular dependency
        from foodspec.validation.nested import NestedCVRunner

        return NestedCVRunner(
            estimator_factory=estimator_factory,
            param_grid=param_grid,
            n_outer_splits=n_splits,
            n_inner_splits=n_inner_splits,
            tuning_metric=tuning_metric,
            seed=seed,
            output_dir=output_dir,
            stability_selector=stability_selector,
            x_wavenumbers=x_wavenumbers,
        )
    else:
        # Use standard CV
        if estimator is None:
            raise ValueError("nested=False requires estimator (pre-configured model)")

        return EvaluationRunner(
            estimator=estimator,
            n_splits=n_splits,
            seed=seed,
            output_dir=output_dir,
            stability_selector=stability_selector,
            x_wavenumbers=x_wavenumbers,
            preprocessors=preprocessors,
            feature_extractors=feature_extractors,
        )


__all__ = ["Estimator", "EvaluationResult", "bootstrap_ci", "EvaluationRunner", "create_evaluation_runner"]
