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
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.features.base import FeatureSet
from foodspec.features.marker_panel import MarkerPanel
from foodspec.trust.abstain import evaluate_abstention
from foodspec.trust.coverage import coverage_by_group
from foodspec.validation.metrics import accuracy, auroc_macro, macro_f1
from foodspec.validation.splits import StratifiedKFoldOrGroupKFold
from foodspec.validation.statistics import bootstrap_ci as compute_bootstrap_ci


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
        Bootstrap confidence intervals: metric -> (lower, median, upper).
    hyperparameters_per_fold : list of dict, optional
        Hyperparameters selected per fold (only for nested CV).
    """

    fold_predictions: List[Dict[str, Any]]  # Per-fold pred dicts: idx, y_true, y_pred, proba...
    fold_metrics: List[Dict[str, float]]  # Per-fold metrics: fold_id, accuracy, macro_f1, auroc
    bootstrap_ci: Dict[str, Tuple[float, float, float]]  # metric -> (lower, median, upper)
    hyperparameters_per_fold: Optional[List[Dict[str, Any]]] = None  # For nested CV

    def save_predictions_csv(self, path: Path) -> None:
        """Save per-sample predictions to CSV.
        
        Columns: fold_id, sample_idx, y_true, y_pred, proba_0, proba_1, ..., group (if present)
        
        Parameters
        ----------
        path : Path
            Output CSV path for predictions.
        
        Examples
        --------
        >>> result.save_predictions_csv(Path("predictions.csv"))
        """
        if not self.fold_predictions:
            Path(path).write_text("")
            return
        
        # Write using ArtifactRegistry helper
        from foodspec.core.artifacts import ArtifactRegistry
        registry = ArtifactRegistry(Path(path).parent)
        registry.write_csv(path, self.fold_predictions)

    def save_metrics_csv(self, path: Path, include_summary: bool = True) -> None:
        """Save per-fold metrics and summary statistics to CSV.
        
        If include_summary=True, appends summary rows with mean, std, and bootstrap CI.
        
        Parameters
        ----------
        path : Path
            Output CSV path for metrics.
        include_summary : bool, default True
            Whether to append summary statistics (mean, std, CI).
        
        Examples
        --------
        >>> result.save_metrics_csv(Path("metrics.csv"))
        """
        if not self.fold_metrics:
            Path(path).write_text("")
            return
        
        rows = list(self.fold_metrics)
        
        if include_summary:
            # Compute summary statistics
            metric_names = [k for k in self.fold_metrics[0].keys() if k != "fold_id"]
            
            # Mean row
            mean_row = {"fold_id": "mean"}
            for metric in metric_names:
                values = [fold[metric] for fold in self.fold_metrics]
                valid_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
                mean_row[metric] = np.mean(valid_values) if valid_values else float('nan')
            rows.append(mean_row)
            
            # Std row
            std_row = {"fold_id": "std"}
            for metric in metric_names:
                values = [fold[metric] for fold in self.fold_metrics]
                valid_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
                std_row[metric] = np.std(valid_values, ddof=1) if len(valid_values) > 1 else float('nan')
            rows.append(std_row)
            
            # Bootstrap CI rows (lower, median, upper)
            ci_lower_row = {"fold_id": "ci_lower"}
            ci_median_row = {"fold_id": "ci_median"}
            ci_upper_row = {"fold_id": "ci_upper"}
            
            for metric in metric_names:
                if metric in self.bootstrap_ci:
                    lower, median, upper = self.bootstrap_ci[metric]
                    ci_lower_row[metric] = lower
                    ci_median_row[metric] = median
                    ci_upper_row[metric] = upper
                else:
                    ci_lower_row[metric] = float('nan')
                    ci_median_row[metric] = float('nan')
                    ci_upper_row[metric] = float('nan')
            
            rows.extend([ci_lower_row, ci_median_row, ci_upper_row])
        
        # Write using ArtifactRegistry helper
        from foodspec.core.artifacts import ArtifactRegistry
        registry = ArtifactRegistry(Path(path).parent)
        registry.write_csv(path, rows)

    def save_best_params_csv(self, path: Path) -> None:
        """Save hyperparameters per fold to CSV (nested CV only).
        
        Columns: fold_id, param1, param2, ...
        
        Parameters
        ----------
        path : Path
            Output CSV path for hyperparameters.
        
        Examples
        --------
        >>> result.save_best_params_csv(Path("best_params.csv"))
        """
        if not self.hyperparameters_per_fold:
            Path(path).write_text("")
            return
        
        # Add fold_id to each parameter dict
        rows = [
            {"fold_id": i, **params}
            for i, params in enumerate(self.hyperparameters_per_fold)
        ]
        
        # Write using ArtifactRegistry helper
        from foodspec.core.artifacts import ArtifactRegistry
        registry = ArtifactRegistry(Path(path).parent)
        registry.write_csv(path, rows)



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

            acc_dict = accuracy(y_test, pred, proba)
            f1_dict = macro_f1(y_test, pred, proba)
            try:
                roc_dict = auroc_macro(y_test, pred, proba)
                roc = roc_dict["auroc_macro"]
            except Exception:
                roc = np.nan

            acc = acc_dict["accuracy"]
            f1 = f1_dict["macro_f1"]

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
            bootstrap_ci_vals["macro_f1"] = compute_bootstrap_ci(valid_f1s, n_boot=1000, alpha=0.05, seed=self.seed)
        if len(valid_aurocs) > 0:
            bootstrap_ci_vals["auroc"] = compute_bootstrap_ci(valid_aurocs, n_boot=1000, alpha=0.05, seed=self.seed)
        bootstrap_ci_vals["accuracy"] = compute_bootstrap_ci(np.array(accuracies), n_boot=1000, alpha=0.05, seed=self.seed)

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


def evaluate_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    splitter: Any,
    feature_extractor: Optional[Any] = None,
    selector: Optional[Any] = None,
    calibrator: Optional[Any] = None,
    calibration_fraction: float = 0.2,
    conformal_calibrator: Optional[Any] = None,
    condition_key: Optional[str] = None,
    abstain_threshold: Optional[float] = None,
    abstain_max_set_size: Optional[int] = None,
    trust_output_dir: Optional[Path | str] = None,
    metrics: Optional[List[str]] = None,
    seed: int = 0,
    meta: Optional[pd.DataFrame] = None,
    x_grid: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """Evaluate a model using cross-validation with leakage-safe pipeline.

    This function implements a complete evaluation pipeline that ensures no data
    leakage occurs between training and test sets. All fitting operations
    (feature extraction, selection, model training, calibration) are performed
    only on training data, with transformations applied to test data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix (e.g., spectra).
    y : np.ndarray, shape (n_samples,)
        Target labels.
    model : BaseEstimator
        Scikit-learn compatible model with fit() and predict_proba() methods.
    splitter : BaseCrossValidator
        Cross-validation splitter (e.g., StratifiedKFold, GroupKFold).
        Must have split(X, y, groups) method.
    feature_extractor : optional
        Feature extractor with fit(X, y) and transform(X) methods.
        If None, raw features are used.
    selector : optional
        Feature selector with fit(X, y) and transform(X) methods.
        If None, all features are used.
    calibrator : optional
        Probability calibrator (e.g., CalibratedClassifierCV).
        If None, raw model probabilities are used.
    metrics : list of str, optional
        Metrics to compute. Options: 'accuracy', 'macro_f1', 'precision_macro',
        'recall_macro', 'auroc_macro', 'ece'.
        If None, defaults to ['accuracy', 'macro_f1', 'auroc_macro'].
    seed : int, default 0
        Random seed for reproducibility.
    meta : pd.DataFrame, optional
        Metadata with 'group' column for group-aware CV.
    x_grid : np.ndarray, optional
        Wavenumber or wavelength grid (for documentation).

    Returns
    -------
    result : EvaluationResult
        Evaluation results containing:
        - fold_predictions: List of per-sample predictions with fold_id and group
        - fold_metrics: List of per-fold metric dictionaries
        - bootstrap_ci: Bootstrap confidence intervals for each metric
        - hyperparameters_per_fold: None (reserved for nested CV)

    Examples
    --------
    Basic usage with raw features:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import StratifiedKFold
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)
    >>> y = np.random.randint(0, 2, 100)
    >>> model = LogisticRegression(random_state=42)
    >>> splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> result = evaluate_model_cv(X, y, model, splitter, seed=42)
    >>> len(result.fold_metrics)
    5
    >>> 'accuracy' in result.fold_metrics[0]
    True

    With feature extraction and selection:

    >>> from foodspec.features.bands import BandIntegrationExtractor
    >>> from sklearn.feature_selection import SelectKBest
    >>> extractor = BandIntegrationExtractor(bands=[(1000, 1100), (1500, 1600)])
    >>> selector = SelectKBest(k=10)
    >>> result = evaluate_model_cv(
    ...     X, y, model, splitter,
    ...     feature_extractor=extractor,
    ...     selector=selector,
    ...     metrics=['accuracy', 'macro_f1', 'auroc_macro'],
    ...     seed=42
    ... )

    Notes
    -----
    - All fitting is done on training folds only (no leakage)
    - Predictions are deterministic given the same seed
    - Group information from meta is used to track which samples were held out
    - Each fold is evaluated independently to ensure fair comparison
    """
    # Input validation
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same length, got {X.shape[0]} and {y.shape[0]}")

    # Set default metrics
    if metrics is None:
        metrics = ["accuracy", "macro_f1", "auroc_macro"]

    # Map metric names to functions
    from foodspec.validation.metrics import (
        accuracy as metric_accuracy,
        macro_f1 as metric_macro_f1,
        precision_macro as metric_precision_macro,
        recall_macro as metric_recall_macro,
        auroc_macro as metric_auroc_macro,
        expected_calibration_error as metric_ece,
    )

    metric_functions = {
        "accuracy": metric_accuracy,
        "macro_f1": metric_macro_f1,
        "precision_macro": metric_precision_macro,
        "recall_macro": metric_recall_macro,
        "auroc_macro": metric_auroc_macro,
        "ece": metric_ece,
    }

    # Optional trust artifacts registry
    trust_registry = None
    if trust_output_dir is not None:
        trust_registry = ArtifactRegistry(Path(trust_output_dir))
        trust_registry.ensure_layout()

    # Pass metadata or groups to splitter
    # The splitter's split() method accepts either:
    # - A DataFrame (new API for metadata-based splitting - our custom splitters)
    # - A groups array (legacy API for sklearn GroupKFold, LeaveOneGroupOut, etc.)
    # - None (for non-group-aware splitting)
    groups_or_meta = None
    if meta is not None:
        # Check if this is one of our custom splitters or an sklearn splitter
        # Our custom splitters are in foodspec.validation.splits module
        splitter_module = splitter.__class__.__module__
        if splitter_module.startswith('foodspec.validation.splits'):
            # Our custom splitters accept DataFrame
            groups_or_meta = meta
        elif "group" in meta.columns:
            # Sklearn splitters expect 1D groups array
            groups_or_meta = meta["group"].values
        else:
            # For sklearn splitters without "group" column, pass None
            groups_or_meta = None
    
    # Storage for results
    fold_predictions = []
    fold_metrics = []
    all_coverage_rows: List[pd.DataFrame] = []
    
    # Track metrics across folds for bootstrap CI
    metrics_per_fold = {metric_name: [] for metric_name in metrics}

    # Iterate through cross-validation folds
    for fold_id, split_result in enumerate(splitter.split(X, y, groups_or_meta)):
        # Handle both old API (2-tuple) and new API (3-tuple with fold_info)
        if len(split_result) == 2:
            train_idx, test_idx = split_result
            fold_info = {}
        else:
            train_idx, test_idx, fold_info = split_result
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Track which group each test sample belongs to (for recording in predictions)
        # Extract group information from metadata if available
        test_groups = None
        if meta is not None:
            # Try common group column names
            for col in ["group", "batch", "stage"]:
                if col in meta.columns:
                    test_groups = meta.iloc[test_idx][col].values
                    break

        # 1. Feature extraction (fit on train only)
        if feature_extractor is not None:
            extractor_fold = copy.deepcopy(feature_extractor)
            # Try to fit with y, fall back to unsupervised if needed
            try:
                extractor_fold.fit(X_train, y_train)
            except TypeError:
                extractor_fold.fit(X_train)
            
            X_train_features = extractor_fold.transform(X_train)
            X_test_features = extractor_fold.transform(X_test)
        else:
            X_train_features = X_train
            X_test_features = X_test

        # Handle FeatureSet objects
        if hasattr(X_train_features, 'Xf'):
            X_train_features = X_train_features.Xf
        if hasattr(X_test_features, 'Xf'):
            X_test_features = X_test_features.Xf

        # Convert to numpy arrays
        X_train_features = np.asarray(X_train_features, dtype=float)
        X_test_features = np.asarray(X_test_features, dtype=float)

        # 2. Feature selection (fit on train only)
        if selector is not None:
            selector_fold = copy.deepcopy(selector)
            selector_fold.fit(X_train_features, y_train)
            X_train_selected = selector_fold.transform(X_train_features)
            X_test_selected = selector_fold.transform(X_test_features)
        else:
            X_train_selected = X_train_features
            X_test_selected = X_test_features

        # 3. Optional calibration split (train -> train_fit + train_cal)
        use_calibration = (calibrator is not None) or (conformal_calibrator is not None)
        if use_calibration:
            if not 0.0 < calibration_fraction < 1.0:
                raise ValueError("calibration_fraction must be in (0, 1) when calibration is enabled")
            rng_seed = seed + fold_id
            if condition_key and meta is not None and condition_key in meta.columns:
                groups_train = meta.iloc[train_idx][condition_key].values
                splitter_cal = GroupShuffleSplit(
                    n_splits=1, test_size=calibration_fraction, random_state=rng_seed
                )
                fit_local, cal_local = next(splitter_cal.split(X_train_selected, y_train, groups=groups_train))
            else:
                splitter_cal = StratifiedShuffleSplit(
                    n_splits=1, test_size=calibration_fraction, random_state=rng_seed
                )
                fit_local, cal_local = next(splitter_cal.split(X_train_selected, y_train))
        else:
            fit_local = np.arange(X_train_selected.shape[0])
            cal_local = np.array([], dtype=int)

        X_fit, y_fit = X_train_selected[fit_local], y_train[fit_local]
        X_cal = X_train_selected[cal_local] if cal_local.size > 0 else None
        y_cal = y_train[cal_local] if cal_local.size > 0 else None

        # 4. Model training (fit on train_fit only)
        model_fold = copy.deepcopy(model)
        model_fold.fit(X_fit, y_fit)

        # 5. Probabilities + calibration on train_cal and test
        proba_test_uncal = model_fold.predict_proba(X_test_selected)
        proba_cal_uncal = model_fold.predict_proba(X_cal) if X_cal is not None else None

        calibrator_fold = copy.deepcopy(calibrator) if calibrator is not None else None

        def _apply_calibrator(cal, proba_array):
            if hasattr(cal, "predict") and not hasattr(cal, "transform"):
                return cal.predict(proba_array)
            if hasattr(cal, "transform"):
                return cal.transform(proba_array)
            if hasattr(cal, "predict_proba"):
                return cal.predict_proba(proba_array)
            raise TypeError("calibrator must implement predict, transform, or predict_proba")

        if calibrator_fold is not None:
            if proba_cal_uncal is None or y_cal is None:
                raise ValueError("Calibration split is empty; increase calibration_fraction")
            calibrator_fold.fit(y_cal, proba_cal_uncal)
            proba_calibrated_cal = _apply_calibrator(calibrator_fold, proba_cal_uncal)
            proba_test = _apply_calibrator(calibrator_fold, proba_test_uncal)
        else:
            proba_calibrated_cal = proba_cal_uncal
            proba_test = proba_test_uncal

        # 5b. Conformal prediction using calibration split only
        cp_result = None
        coverage_df_fold = None
        if conformal_calibrator is not None:
            if proba_calibrated_cal is None or y_cal is None:
                raise ValueError("conformal_calibrator requires a non-empty calibration split")
            cp_fold = copy.deepcopy(conformal_calibrator)

            meta_cal_vals = None
            meta_test_vals = None
            if condition_key and meta is not None and condition_key in meta.columns:
                meta_train_df = meta.iloc[train_idx].reset_index(drop=True)
                meta_cal_vals = meta_train_df[condition_key].values[cal_local]
                meta_test_vals = meta.iloc[test_idx][condition_key].values

            cp_fold.fit(y_cal, proba_calibrated_cal, meta_cal=meta_cal_vals)
            cp_result = cp_fold.predict_sets(proba_test, meta_test=meta_test_vals, y_true=y_test)

            # Build coverage dataframe (ensure bin column exists)
            bin_values = meta_test_vals if meta_test_vals is not None else np.array(["global"] * len(y_test))
            coverage_df_fold = cp_result.to_dataframe(y_true=y_test, bin_values=bin_values)
            coverage_df_fold["sample_idx"] = test_idx
            coverage_df_fold["fold_id"] = fold_id
            all_coverage_rows.append(coverage_df_fold)

        # 6. Predictions
        if proba_test.ndim == 2:
            y_pred = np.argmax(proba_test, axis=1)
        else:
            # Binary case with 1D probabilities
            y_pred = (proba_test > 0.5).astype(int)

        # 6. Compute metrics for this fold
        fold_metric_dict = {"fold_id": fold_id}
        
        for metric_name in metrics:
            if metric_name not in metric_functions:
                raise ValueError(f"Unknown metric: {metric_name}. "
                               f"Available: {list(metric_functions.keys())}")
            
            metric_func = metric_functions[metric_name]
            
            try:
                # Call metric function
                if metric_name == "ece":
                    # ECE has n_bins parameter
                    metric_result = metric_func(y_test, y_pred, proba_test, n_bins=10)
                else:
                    metric_result = metric_func(y_test, y_pred, proba_test)
                
                # Extract scalar value from dict
                metric_key = list(metric_result.keys())[0]
                metric_value = metric_result[metric_key]
                
                fold_metric_dict[metric_name] = metric_value
                metrics_per_fold[metric_name].append(metric_value)
                
            except Exception as e:
                # Handle cases where metric can't be computed (e.g., single class)
                fold_metric_dict[metric_name] = float('nan')
                metrics_per_fold[metric_name].append(float('nan'))

        # Add trust metrics if available
        if cp_result is not None:
            fold_metric_dict["coverage"] = cp_result.coverage
            metrics_per_fold.setdefault("coverage", []).append(cp_result.coverage)

        abstention_summary = None
        if abstain_threshold is not None:
            prediction_sets_for_abstain = cp_result.prediction_sets if cp_result is not None else None
            max_set_size = abstain_max_set_size if prediction_sets_for_abstain is not None else None
            abstention = evaluate_abstention(
                proba_test,
                y_test,
                threshold=abstain_threshold,
                prediction_sets=prediction_sets_for_abstain,
                max_set_size=max_set_size,
            )
            abstention_summary = {
                "fold_id": fold_id,
                "abstain_rate": abstention.abstain_rate,
                "accuracy_on_answered": abstention.accuracy_non_abstained,
                "coverage_on_answered": abstention.coverage,
            }
            metrics_per_fold.setdefault("abstain_rate", []).append(abstention.abstain_rate)

        fold_metrics.append(fold_metric_dict)

        # 7. Store per-sample predictions
        for i, sample_idx in enumerate(test_idx):
            pred_dict = {
                "fold_id": fold_id,
                "sample_idx": int(sample_idx),
                "y_true": int(y_test[i]),
                "y_pred": int(y_pred[i]),
            }
            
            # Add probabilities
            if proba_test.ndim == 2:
                for class_idx in range(proba_test.shape[1]):
                    pred_dict[f"proba_{class_idx}"] = float(proba_test[i, class_idx])
            else:
                pred_dict["proba_0"] = 1.0 - float(proba_test[i])
                pred_dict["proba_1"] = float(proba_test[i])
            
            # Add group if available
            if test_groups is not None:
                pred_dict["group"] = test_groups[i]
            
            fold_predictions.append(pred_dict)

        # Save trust artifacts per fold (if requested)
        if trust_registry is not None:
            trust_dir = trust_registry.trust_dir
            # Calibration probabilities
            if proba_calibrated_cal is not None and y_cal is not None and cal_local.size > 0:
                cal_rows = []
                cal_sample_idx = train_idx[cal_local]
                for j, idx_cal in enumerate(cal_sample_idx):
                    row = {"sample_idx": int(idx_cal)}
                    for class_idx in range(proba_calibrated_cal.shape[1]):
                        row[f"proba_{class_idx}"] = float(proba_calibrated_cal[j, class_idx])
                    cal_rows.append(row)
                trust_registry.write_csv(trust_dir / f"calibrated_proba_fold_{fold_id}.csv", cal_rows)

            # Test probabilities
            test_rows = []
            for j, idx_test in enumerate(test_idx):
                row = {"sample_idx": int(idx_test)}
                for class_idx in range(proba_test.shape[1] if proba_test.ndim == 2 else 2):
                    if proba_test.ndim == 2:
                        row[f"proba_{class_idx}"] = float(proba_test[j, class_idx])
                    else:
                        if class_idx == 0:
                            row[f"proba_{class_idx}"] = 1.0 - float(proba_test[j])
                        else:
                            row[f"proba_{class_idx}"] = float(proba_test[j])
                test_rows.append(row)
            trust_registry.write_csv(trust_dir / f"calibrated_test_proba_fold_{fold_id}.csv", test_rows)

            # Conformal sets and coverage
            if coverage_df_fold is not None:
                trust_registry.write_csv(trust_dir / f"conformal_sets_fold_{fold_id}.csv", coverage_df_fold.to_dict(orient="records"))
                coverage_table_fold = coverage_by_group(coverage_df_fold, group_col="bin")
                trust_registry.write_csv(trust_dir / f"coverage_fold_{fold_id}.csv", coverage_table_fold.to_dict(orient="records"))

            # Abstention summary
            if abstention_summary is not None:
                trust_registry.write_csv(trust_dir / f"abstention_fold_{fold_id}.csv", [abstention_summary])

    # 8. Compute bootstrap confidence intervals
    bootstrap_ci_dict = {}
    for metric_name in metrics_per_fold.keys():
        values = np.array(metrics_per_fold[metric_name])
        # Remove NaN values before computing CI
        values_valid = values[~np.isnan(values)]
        if len(values_valid) > 0:
            ci_lower, ci_median, ci_upper = compute_bootstrap_ci(
                values_valid, n_boot=1000, alpha=0.05, seed=seed
            )
            bootstrap_ci_dict[metric_name] = (ci_lower, ci_median, ci_upper)
        else:
            bootstrap_ci_dict[metric_name] = (float('nan'), float('nan'), float('nan'))

    # Aggregate coverage across folds if available
    if trust_registry is not None and all_coverage_rows:
        combined_coverage = pd.concat(all_coverage_rows, ignore_index=True)
        coverage_table_all = coverage_by_group(combined_coverage, group_col="bin")
        trust_registry.write_csv(trust_registry.trust_dir / "coverage_overall.csv", coverage_table_all.to_dict(orient="records"))

    # 9. Return results
    return EvaluationResult(
        fold_predictions=fold_predictions,
        fold_metrics=fold_metrics,
        bootstrap_ci=bootstrap_ci_dict,
        hyperparameters_per_fold=None,
    )


def evaluate_model_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Any,
    outer_splitter: Any,
    inner_splitter: Any,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    param_distributions: Optional[Dict[str, Any]] = None,
    search_strategy: str = "grid",
    feature_extractor: Optional[Any] = None,
    selector: Optional[Any] = None,
    calibrator: Optional[Any] = None,
    metrics: Optional[List[str]] = None,
    tuning_metric: str = "macro_f1",
    seed: int = 0,
    meta: Optional[pd.DataFrame] = None,
    x_grid: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """Nested cross-validation with hyperparameter tuning (inner CV) and unbiased evaluation (outer CV).

    This function implements nested CV to obtain unbiased performance estimates while tuning
    hyperparameters. The outer loop provides test folds for unbiased evaluation, while the
    inner loop (within each outer training fold) is used to select optimal hyperparameters.

    All components are fitted only on the outer training set during each outer fold to prevent
    data leakage. The inner CV is used strictly for hyperparameter selection within the outer
    training fold.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix (e.g., spectra).
    y : np.ndarray, shape (n_samples,)
        Target labels.
    model_factory : callable
        Factory function that takes hyperparameters as kwargs and returns an unfitted model.
        Example: lambda C=1.0, max_iter=100: LogisticRegression(C=C, max_iter=max_iter)
    outer_splitter : BaseCrossValidator
        Cross-validation splitter for outer loop (e.g., GroupKFold for LOBO).
        Must have split(X, y, groups) method.
    inner_splitter : BaseCrossValidator
        Cross-validation splitter for inner loop (hyperparameter search).
        Must have split(X, y, groups) method.
    param_grid : dict, optional
        Dictionary mapping parameter names to lists of values for grid search.
        Used if search_strategy='grid'. Example: {'C': [0.1, 1.0, 10.0]}
    param_distributions : dict, optional
        Dictionary mapping parameter names to scipy.stats distributions for random search.
        Used if search_strategy='randomized'.
    search_strategy : {'grid', 'randomized'}, default 'grid'
        Hyperparameter search strategy. 'grid' exhaustively searches param_grid.
        'randomized' samples from param_distributions.
    feature_extractor : optional
        Feature extractor with fit(X, y) and transform(X) methods.
        Fitted only on outer training set per fold.
    selector : optional
        Feature selector with fit(X, y) and transform(X) methods.
        Fitted only on outer training set per fold.
    calibrator : optional
        Probability calibrator (e.g., CalibratedClassifierCV).
        Fitted only on outer training set per fold.
    metrics : list of str, optional
        Metrics to compute. Options: 'accuracy', 'macro_f1', 'precision_macro',
        'recall_macro', 'auroc_macro', 'ece'.
        If None, defaults to ['accuracy', 'macro_f1', 'auroc_macro'].
    tuning_metric : str, default 'macro_f1'
        Metric to optimize during inner CV hyperparameter search.
        Options: 'accuracy', 'macro_f1', 'auroc_macro'
    seed : int, default 0
        Random seed for reproducibility.
    meta : pd.DataFrame, optional
        Metadata with 'group' column for group-aware CV.
    x_grid : np.ndarray, optional
        Wavenumber or wavelength grid (for documentation).

    Returns
    -------
    result : EvaluationResult
        Evaluation results containing:
        - fold_predictions: Per-sample predictions from outer test sets
        - fold_metrics: Per-fold metrics on outer test sets
        - bootstrap_ci: Bootstrap confidence intervals for each metric
        - hyperparameters_per_fold: Hyperparameters selected per outer fold

    Raises
    ------
    ValueError
        If X and y shapes don't match, if search_strategy is invalid, or if metric names are unknown.

    Notes
    -----
    - **Outer loop**: Provides test sets for unbiased evaluation
    - **Inner loop**: Runs within each outer training fold to select best hyperparameters
    - All fitting done on outer training only (strict leakage prevention)
    - Hyperparameters selected based on tuning_metric via inner CV
    - Final model per outer fold trained on full outer training set with best hyperparameters
    - Evaluation metrics computed only on outer test sets (unbiased)

    Examples
    --------
    Grid search with LogisticRegression:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import StratifiedKFold, GroupKFold
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)
    >>> y = np.random.randint(0, 2, 100)
    >>> model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=200)
    >>> outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    >>> inner_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    >>> result = evaluate_model_nested_cv(
    ...     X, y, model_factory, outer_splitter, inner_splitter,
    ...     param_grid={'C': [0.1, 1.0, 10.0]},
    ...     search_strategy='grid',
    ...     seed=42
    ... )
    >>> len(result.hyperparameters_per_fold)
    3
    >>> result.hyperparameters_per_fold[0]['C'] in [0.1, 1.0, 10.0]
    True
    """
    # Input validation
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same length, got {X.shape[0]} and {y.shape[0]}")

    if search_strategy not in ("grid", "randomized"):
        raise ValueError(f"search_strategy must be 'grid' or 'randomized', got {search_strategy}")

    # Set default metrics
    if metrics is None:
        metrics = ["accuracy", "macro_f1", "auroc_macro"]

    # Map metric names to functions
    from foodspec.validation.metrics import (
        accuracy as metric_accuracy,
        macro_f1 as metric_macro_f1,
        precision_macro as metric_precision_macro,
        recall_macro as metric_recall_macro,
        auroc_macro as metric_auroc_macro,
        expected_calibration_error as metric_ece,
    )

    metric_functions = {
        "accuracy": metric_accuracy,
        "macro_f1": metric_macro_f1,
        "precision_macro": metric_precision_macro,
        "recall_macro": metric_recall_macro,
        "auroc_macro": metric_auroc_macro,
        "ece": metric_ece,
    }

    # Validate tuning metric
    if tuning_metric not in metric_functions:
        raise ValueError(f"tuning_metric '{tuning_metric}' not in {list(metric_functions.keys())}")

    # Pass metadata or groups to splitter
    groups_or_meta = None
    if meta is not None:
        # Check if this is one of our custom splitters or an sklearn splitter
        # Our custom splitters are in foodspec.validation.splits module
        splitter_module = outer_splitter.__class__.__module__
        if splitter_module.startswith('foodspec.validation.splits'):
            # Our custom splitters accept DataFrame
            groups_or_meta = meta
        elif "group" in meta.columns:
            # Sklearn splitters expect 1D groups array
            groups_or_meta = meta["group"].values
        else:
            # For sklearn splitters without "group" column, pass None
            groups_or_meta = None

    # Storage for results
    fold_predictions = []
    fold_metrics = []
    hyperparameters_per_fold = []
    
    # Track metrics across folds for bootstrap CI
    metrics_per_fold = {metric_name: [] for metric_name in metrics}

    # Outer CV loop
    for outer_fold_id, outer_split_result in enumerate(outer_splitter.split(X, y, groups_or_meta)):
        # Handle both old API (2-tuple) and new API (3-tuple with fold_info)
        if len(outer_split_result) == 2:
            outer_train_idx, outer_test_idx = outer_split_result
            outer_fold_info = {}
        else:
            outer_train_idx, outer_test_idx, outer_fold_info = outer_split_result
        
        # Split outer fold
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]

        # Extract metadata subset for inner CV
        outer_train_meta = None
        if meta is not None:
            outer_train_meta = meta.iloc[outer_train_idx].reset_index(drop=True)

        # Get test groups for prediction tracking
        outer_test_groups = None
        if meta is not None:
            # Try common group column names
            for col in ["group", "batch", "stage"]:
                if col in meta.columns:
                    outer_test_groups = meta.iloc[outer_test_idx][col].values
                    break

        # ==== INNER CV: Hyperparameter Selection ====
        best_params = _select_hyperparameters_nested_cv(
            X_outer_train,
            y_outer_train,
            model_factory,
            inner_splitter,
            param_grid,
            param_distributions,
            search_strategy,
            feature_extractor,
            selector,
            tuning_metric,
            metric_functions,
            outer_train_meta,
            seed,
        )

        hyperparameters_per_fold.append(best_params)

        # ==== OUTER EVALUATION: Train final model with best params ====
        # 1. Feature extraction (fit on outer train only)
        if feature_extractor is not None:
            extractor_fold = copy.deepcopy(feature_extractor)
            try:
                extractor_fold.fit(X_outer_train, y_outer_train)
            except TypeError:
                extractor_fold.fit(X_outer_train)
            
            X_outer_train_features = extractor_fold.transform(X_outer_train)
            X_outer_test_features = extractor_fold.transform(X_outer_test)
        else:
            X_outer_train_features = X_outer_train
            X_outer_test_features = X_outer_test

        # Handle FeatureSet objects
        if hasattr(X_outer_train_features, 'Xf'):
            X_outer_train_features = X_outer_train_features.Xf
        if hasattr(X_outer_test_features, 'Xf'):
            X_outer_test_features = X_outer_test_features.Xf

        # Convert to numpy arrays
        X_outer_train_features = np.asarray(X_outer_train_features, dtype=float)
        X_outer_test_features = np.asarray(X_outer_test_features, dtype=float)

        # 2. Feature selection (fit on outer train only)
        if selector is not None:
            selector_fold = copy.deepcopy(selector)
            selector_fold.fit(X_outer_train_features, y_outer_train)
            X_outer_train_selected = selector_fold.transform(X_outer_train_features)
            X_outer_test_selected = selector_fold.transform(X_outer_test_features)
        else:
            X_outer_train_selected = X_outer_train_features
            X_outer_test_selected = X_outer_test_features

        # 3. Model training (fit on outer train only with best hyperparameters)
        model_fold = model_factory(**best_params)
        model_fold.fit(X_outer_train_selected, y_outer_train)

        # 4. Calibration (fit on outer train only)
        if calibrator is not None:
            proba_outer_train_uncal = model_fold.predict_proba(X_outer_train_selected)
            calibrator_fold = copy.deepcopy(calibrator)
            calibrator_fold.fit(proba_outer_train_uncal, y_outer_train)
            proba_outer_test_uncal = model_fold.predict_proba(X_outer_test_selected)
            # Use predict_proba if available (CalibratedClassifierCV), else transform
            if hasattr(calibrator_fold, 'predict_proba'):
                proba_outer_test = calibrator_fold.predict_proba(X_outer_test_selected)
            else:
                proba_outer_test = calibrator_fold.transform(proba_outer_test_uncal)
        else:
            proba_outer_test = model_fold.predict_proba(X_outer_test_selected)

        # 5. Make predictions on outer test
        y_pred_outer_test = proba_outer_test.argmax(axis=1)

        # 6. Compute metrics on outer test
        fold_metrics_dict = {"fold_id": outer_fold_id}
        for metric_name in metrics:
            if metric_name not in metric_functions:
                raise ValueError(f"Unknown metric: {metric_name}")
            metric_func = metric_functions[metric_name]
            metric_result = metric_func(y_outer_test, y_pred_outer_test, proba_outer_test)
            metric_value = metric_result[metric_name]
            fold_metrics_dict[metric_name] = metric_value
            metrics_per_fold[metric_name].append(metric_value)

        fold_metrics.append(fold_metrics_dict)

        # 7. Store per-sample predictions from outer test
        for i, sample_idx in enumerate(outer_test_idx):
            pred_dict = {
                "fold_id": outer_fold_id,
                "sample_idx": int(sample_idx),
                "y_true": int(y_outer_test[i]),
                "y_pred": int(y_pred_outer_test[i]),
            }

            # Store probabilities for each class
            if proba_outer_test.ndim == 2:
                for class_idx in range(proba_outer_test.shape[1]):
                    pred_dict[f"proba_{class_idx}"] = float(proba_outer_test[i, class_idx])
            else:
                pred_dict["proba_0"] = 1.0 - float(proba_outer_test[i])
                pred_dict["proba_1"] = float(proba_outer_test[i])

            # Add group if available
            if outer_test_groups is not None:
                pred_dict["group"] = outer_test_groups[i]

            fold_predictions.append(pred_dict)

    # Compute bootstrap confidence intervals
    bootstrap_ci_dict = {}
    for metric_name in metrics:
        values = np.array(metrics_per_fold[metric_name])
        values_valid = values[~np.isnan(values)]
        if len(values_valid) > 0:
            ci_lower, ci_median, ci_upper = compute_bootstrap_ci(
                values_valid, n_boot=1000, alpha=0.05, seed=seed
            )
            bootstrap_ci_dict[metric_name] = (ci_lower, ci_median, ci_upper)
        else:
            bootstrap_ci_dict[metric_name] = (float('nan'), float('nan'), float('nan'))

    # Return results
    return EvaluationResult(
        fold_predictions=fold_predictions,
        fold_metrics=fold_metrics,
        bootstrap_ci=bootstrap_ci_dict,
        hyperparameters_per_fold=hyperparameters_per_fold,
    )


def _select_hyperparameters_nested_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_factory: Any,
    inner_splitter: Any,
    param_grid: Optional[Dict[str, List[Any]]],
    param_distributions: Optional[Dict[str, Any]],
    search_strategy: str,
    feature_extractor: Optional[Any],
    selector: Optional[Any],
    tuning_metric: str,
    metric_functions: Dict[str, Callable],
    meta_train: Optional[pd.DataFrame],
    seed: int,
) -> Dict[str, Any]:
    """Select best hyperparameters using inner CV.

    This is a helper function that runs hyperparameter search within the outer training fold.
    It ensures strict leakage avoidance by only using the outer training data.

    Parameters
    ----------
    X_train : np.ndarray
        Outer training features.
    y_train : np.ndarray
        Outer training labels.
    model_factory : callable
        Function that creates models with hyperparameters.
    inner_splitter : BaseCrossValidator
        Inner CV splitter for hyperparameter evaluation.
    param_grid : dict, optional
        Parameter grid for grid search.
    param_distributions : dict, optional
        Parameter distributions for random search.
    search_strategy : str
        'grid' or 'randomized'.
    feature_extractor : optional
        Feature extractor for inner CV.
    selector : optional
        Feature selector for inner CV.
    tuning_metric : str
        Metric to optimize.
    metric_functions : dict
        Metric name to function mapping.
    meta_train : pd.DataFrame, optional
        Metadata for grouped CV.
    seed : int
        Random seed.

    Returns
    -------
    best_params : dict
        Selected hyperparameters.
    """
    # Generate parameter combinations to try
    if search_strategy == "grid":
        if not param_grid:
            return {}  # No hyperparameters to tune
        param_combinations = _generate_param_combinations(param_grid)
    elif search_strategy == "randomized":
        if not param_distributions:
            return {}
        rng = np.random.default_rng(seed)
        n_iter = 10  # Number of random iterations
        param_combinations = _generate_random_params(param_distributions, n_iter, rng)
    else:
        raise ValueError(f"Unknown search_strategy: {search_strategy}")

    if not param_combinations:
        return {}

    # Evaluate each parameter combination via inner CV
    best_score = -np.inf
    best_params = param_combinations[0]

    # Determine what to pass to inner splitter
    inner_groups_or_meta = None
    if meta_train is not None:
        splitter_module = inner_splitter.__class__.__module__
        if splitter_module.startswith('foodspec.validation.splits'):
            # Our custom splitters accept DataFrame
            inner_groups_or_meta = meta_train
        elif "group" in meta_train.columns:
            # Sklearn splitters expect 1D groups array
            inner_groups_or_meta = meta_train["group"].values
        else:
            inner_groups_or_meta = None

    for params in param_combinations:
        scores = []

        # Inner CV loop within outer training fold
        for inner_split_result in inner_splitter.split(X_train, y_train, inner_groups_or_meta):
            # Handle both old API (2-tuple) and new API (3-tuple with fold_info)
            if len(inner_split_result) == 2:
                inner_train_idx, inner_val_idx = inner_split_result
            else:
                inner_train_idx, inner_val_idx, _ = inner_split_result
            
            X_inner_train = X_train[inner_train_idx]
            X_inner_val = X_train[inner_val_idx]
            y_inner_train = y_train[inner_train_idx]
            y_inner_val = y_train[inner_val_idx]

            # Apply feature extraction (fit on inner train only)
            if feature_extractor is not None:
                extractor_inner = copy.deepcopy(feature_extractor)
                try:
                    extractor_inner.fit(X_inner_train, y_inner_train)
                except TypeError:
                    extractor_inner.fit(X_inner_train)
                X_inner_train_features = extractor_inner.transform(X_inner_train)
                X_inner_val_features = extractor_inner.transform(X_inner_val)
            else:
                X_inner_train_features = X_inner_train
                X_inner_val_features = X_inner_val

            # Handle FeatureSet objects
            if hasattr(X_inner_train_features, 'Xf'):
                X_inner_train_features = X_inner_train_features.Xf
            if hasattr(X_inner_val_features, 'Xf'):
                X_inner_val_features = X_inner_val_features.Xf

            X_inner_train_features = np.asarray(X_inner_train_features, dtype=float)
            X_inner_val_features = np.asarray(X_inner_val_features, dtype=float)

            # Apply feature selection (fit on inner train only)
            if selector is not None:
                selector_inner = copy.deepcopy(selector)
                selector_inner.fit(X_inner_train_features, y_inner_train)
                X_inner_train_selected = selector_inner.transform(X_inner_train_features)
                X_inner_val_selected = selector_inner.transform(X_inner_val_features)
            else:
                X_inner_train_selected = X_inner_train_features
                X_inner_val_selected = X_inner_val_features

            # Train model with current hyperparameters (fit on inner train only)
            model_inner = model_factory(**params)
            model_inner.fit(X_inner_train_selected, y_inner_train)

            # Predict on inner validation
            proba_inner_val = model_inner.predict_proba(X_inner_val_selected)
            y_pred_inner_val = proba_inner_val.argmax(axis=1)

            # Compute tuning metric
            metric_func = metric_functions[tuning_metric]
            metric_result = metric_func(y_inner_val, y_pred_inner_val, proba_inner_val)
            score = metric_result[tuning_metric]
            scores.append(score)

        # Average score across inner CV folds
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_params


def _generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from grid (Cartesian product)."""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    combinations = []
    _recursive_param_combinations(keys, values, 0, {}, combinations)
    return combinations


def _recursive_param_combinations(
    keys: List[str],
    values: List[List[Any]],
    idx: int,
    current: Dict[str, Any],
    result: List[Dict[str, Any]],
) -> None:
    """Recursively generate all parameter combinations."""
    if idx == len(keys):
        result.append(current.copy())
        return

    for val in values[idx]:
        current[keys[idx]] = val
        _recursive_param_combinations(keys, values, idx + 1, current, result)
        del current[keys[idx]]


def _generate_random_params(
    param_distributions: Dict[str, Any], n_iter: int, rng: np.random.Generator
) -> List[Dict[str, Any]]:
    """Generate random parameter combinations from distributions."""
    combinations = []
    for _ in range(n_iter):
        params = {}
        for param_name, param_dist in param_distributions.items():
            # Assume param_dist has rvs() method (scipy.stats distribution)
            params[param_name] = param_dist.rvs(random_state=rng)
        combinations.append(params)
    return combinations


__all__ = [
    "Estimator",
    "EvaluationResult",
    "bootstrap_ci",
    "EvaluationRunner",
    "create_evaluation_runner",
    "evaluate_model_cv",
    "evaluate_model_nested_cv",
]
