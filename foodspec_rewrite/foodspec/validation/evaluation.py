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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.validation.metrics import accuracy, auroc_ovr, macro_f1
from foodspec.validation.splits import StratifiedKFoldOrGroupKFold


class Estimator(Protocol):
    """Protocol for a scikit-learn-like estimator."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class EvaluationResult:
    """Holds per-fold predictions and metrics, plus bootstrap CIs."""

    fold_predictions: List[Dict[str, Any]]  # Per-fold pred dicts: idx, y_true, y_pred, proba...
    fold_metrics: List[Dict[str, float]]  # Per-fold metrics: fold_id, accuracy, macro_f1, auroc
    bootstrap_ci: Dict[str, Tuple[float, float]]  # metric -> (lower, upper) CI bounds


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

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit and predict
            self.estimator.fit(X_train, y_train)
            proba = self.estimator.predict_proba(X_test)
            pred = proba.argmax(axis=1) if proba.ndim == 2 else (proba > 0.5).astype(int)

            # Compute metrics
            acc = accuracy(y_test, proba)
            f1 = macro_f1(y_test, proba)
            try:
                roc = auroc_ovr(y_test, proba)
            except Exception:
                roc = np.nan

            accuracies.append(acc)
            macro_f1s.append(f1)
            aurocs.append(roc)

            # Store per-fold results
            metrics_row = {"fold_id": fold_id, "accuracy": acc, "macro_f1": f1, "auroc": roc}
            fold_metrics.append(metrics_row)

            # Per-sample predictions
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

        # Bootstrap CIs for metrics
        bootstrap_ci_vals = {}
        valid_f1s = np.array([f for f in macro_f1s if not np.isnan(f)])
        valid_aurocs = np.array([r for r in aurocs if not np.isnan(r)])
        if len(valid_f1s) > 0:
            bootstrap_ci_vals["macro_f1"] = bootstrap_ci(valid_f1s, seed=self.seed)
        if len(valid_aurocs) > 0:
            bootstrap_ci_vals["auroc"] = bootstrap_ci(valid_aurocs, seed=self.seed)
        bootstrap_ci_vals["accuracy"] = bootstrap_ci(np.array(accuracies), seed=self.seed)

        # Save artifacts
        if artifacts:
            artifacts.write_csv(artifacts.predictions_path, fold_predictions)
            artifacts.write_csv(artifacts.metrics_path, fold_metrics)

        return EvaluationResult(
            fold_predictions=fold_predictions,
            fold_metrics=fold_metrics,
            bootstrap_ci=bootstrap_ci_vals,
        )


__all__ = ["Estimator", "EvaluationResult", "bootstrap_ci", "EvaluationRunner"]
