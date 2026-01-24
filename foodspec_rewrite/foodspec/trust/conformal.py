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

Mondrian conformal prediction for multiclass classification with bin conditioning.

The implementation follows the split-conformal recipe: fit on train, compute
nonconformity scores on a held-out calibration split, and build prediction sets
that achieve target coverage. Conditioning by metadata bins (e.g., stage or
instrument) yields a Mondrian variant with per-bin thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def _mondrian_quantile(scores: np.ndarray, coverage: float) -> float:
    """Quantile with conformal finite-sample correction.

    The threshold is the k-th smallest score where k = ceil((n + 1) * (1 - coverage)).
    This guarantees marginal coverage of at least ``coverage``.
    """

    vals = np.sort(np.asarray(scores, dtype=float))
    n = vals.size
    if n == 0:
        raise ValueError("Cannot compute threshold with no calibration scores")
    alpha = 1.0 - coverage
    k = int(np.ceil((n + 1) * alpha))
    k = max(1, min(k, n))
    return float(vals[k - 1])


def _to_bin_key(value: object | None) -> str:
    return "__global__" if value is None else str(value)


@dataclass
class ConformalPredictionResult:
    """Prediction sets and coverage diagnostics.

    Attributes
    ----------
    prediction_sets : list of list of int
        Classes included for each sample, sorted by descending probability.
    set_sizes : list of int
        Size of each prediction set.
    sample_thresholds : list of float
        Threshold applied to each sample (bin-specific or global).
    thresholds : dict
        Mapping of ``"global"`` and bin keys to thresholds.
    coverage : float | None
        Observed coverage if ``y_true`` was provided, else ``None``.
    per_bin_coverage : dict
        Observed coverage per bin if labels provided.
    per_bin_set_size : dict
        Mean set size per bin.
    """

    prediction_sets: List[List[int]]
    set_sizes: List[int]
    sample_thresholds: List[float]
    thresholds: Dict[str, float]
    coverage: float | None
    per_bin_coverage: Dict[str, float]
    per_bin_set_size: Dict[str, float]


class MondrianConformalClassifier:
    """Split-conformal predictor with optional bin conditioning.

    Parameters
    ----------
    estimator : object
        Fitted-capable estimator exposing ``fit`` and ``predict_proba``.
    target_coverage : float, default 0.9
        Desired marginal coverage (e.g., 0.9 -> 90%).

    Examples
    --------
    >>> from foodspec.models import LogisticRegressionClassifier
    >>> import numpy as np
    >>> X_train = np.random.randn(80, 4)
    >>> y_train = np.random.randint(0, 3, 80)
    >>> X_cal = np.random.randn(30, 4)
    >>> y_cal = np.random.randint(0, 3, 30)
    >>> model = LogisticRegressionClassifier(random_state=0, multi_class="multinomial")
    >>> cp = MondrianConformalClassifier(model, target_coverage=0.9)
    >>> cp.fit(X_train, y_train)
    MondrianConformalClassifier(...)
    >>> cp.calibrate(X_cal, y_cal)
    >>> result = cp.predict_sets(X_cal, y_true=y_cal)
    >>> 0.8 <= result.coverage <= 1.0
    True
    """

    def __init__(self, estimator: object, target_coverage: float = 0.9) -> None:
        if not 0.0 < target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0, 1]")
        self.estimator = estimator
        self.target_coverage = target_coverage
        self._thresholds: Dict[str, float] = {}
        self._fitted = False
        self._n_classes: int | None = None
        self._classes: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MondrianConformalClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")
        self.estimator.fit(X, y)
        self._fitted = True
        classes_attr = getattr(self.estimator, "classes_", None)
        if classes_attr is not None:
            self._classes = np.asarray(classes_attr)
            self._n_classes = int(len(self._classes))
        else:
            self._n_classes = int(len(np.unique(y)))
        return self

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        bins: Sequence[object] | None = None,
    ) -> None:
        """Compute nonconformity thresholds on calibration data."""

        if not self._fitted:
            raise RuntimeError("Estimator not fitted; call fit() first.")
        X_cal = np.asarray(X_cal, dtype=float)
        y_cal = np.asarray(y_cal)
        if X_cal.shape[0] != y_cal.shape[0]:
            raise ValueError("Calibration X and y must align")
        proba_cal = self.estimator.predict_proba(X_cal)
        if proba_cal.ndim != 2:
            raise ValueError("predict_proba must return 2D array")
        if self._n_classes and proba_cal.shape[1] != self._n_classes:
            raise ValueError("Calibration probabilities class dimension mismatch")

        if self._classes is not None:
            index_map = {label: idx for idx, label in enumerate(self._classes)}
            try:
                y_indices = np.array([index_map[label] for label in y_cal])
            except KeyError as exc:  # pragma: no cover - guardrail for unseen labels
                raise ValueError("Calibration labels contain unseen class") from exc
        else:
            y_indices = y_cal.astype(int)
        if (y_indices < 0).any() or (y_indices >= proba_cal.shape[1]).any():
            raise ValueError("Calibration labels out of range for probability columns")

        scores = 1.0 - proba_cal[np.arange(proba_cal.shape[0]), y_indices]
        bin_keys = [_to_bin_key(b) for b in bins] if bins is not None else None

        # Global threshold
        self._thresholds = {"global": _mondrian_quantile(scores, self.target_coverage)}

        # Per-bin thresholds when bins provided
        if bin_keys is not None:
            for key in np.unique(bin_keys):
                bin_scores = scores[np.array(bin_keys) == key]
                if bin_scores.size == 0:
                    continue
                self._thresholds[key] = _mondrian_quantile(bin_scores, self.target_coverage)

    def predict_sets(
        self,
        X: np.ndarray,
        bins: Sequence[object] | None = None,
        y_true: np.ndarray | None = None,
    ) -> ConformalPredictionResult:
        """Generate prediction sets; optional labels compute empirical coverage."""

        if not self._thresholds:
            raise RuntimeError("Model not calibrated; call calibrate() first.")
        X = np.asarray(X, dtype=float)
        proba = self.estimator.predict_proba(X)
        return self.predict_sets_from_proba(proba, bins=bins, y_true=y_true)

    def predict_sets_from_proba(
        self,
        proba: np.ndarray,
        bins: Sequence[object] | None = None,
        y_true: np.ndarray | None = None,
    ) -> ConformalPredictionResult:
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError("proba must be 2D (n_samples, n_classes)")
        n_samples, n_classes = proba.shape
        if self._n_classes and n_classes != self._n_classes:
            raise ValueError("Probability class dimension mismatch from calibration")

        bin_keys = [_to_bin_key(b) for b in bins] if bins is not None else ["__global__"] * n_samples
        if len(bin_keys) != n_samples:
            raise ValueError("bins length must match number of samples")

        thresholds = []
        prediction_sets: List[List[int]] = []
        set_sizes: List[int] = []
        for i in range(n_samples):
            key = bin_keys[i]
            t = self._thresholds.get(key, self._thresholds.get("global"))
            if t is None:
                raise RuntimeError("No threshold available for prediction")
            thresholds.append(float(t))
            # Include classes with nonconformity <= t -> prob >= 1 - t
            keep = [int(cls) for cls in np.argsort(-proba[i]) if proba[i, cls] >= 1.0 - t]
            prediction_sets.append(keep)
            set_sizes.append(len(keep))

        coverage = None
        per_bin_coverage: Dict[str, float] = {}
        per_bin_set_size: Dict[str, float] = {}
        if y_true is not None:
            y_true = np.asarray(y_true)
            if y_true.shape[0] != n_samples:
                raise ValueError("y_true length must match predictions")
            covered = np.array([
                int(y_true[i] in prediction_sets[i]) for i in range(n_samples)
            ], dtype=float)
            coverage = float(np.mean(covered))
            unique_bins = np.unique(bin_keys)
            for key in unique_bins:
                mask = np.array(bin_keys) == key
                per_bin_coverage[key] = float(np.mean(covered[mask])) if mask.any() else np.nan
                per_bin_set_size[key] = float(np.mean(np.array(set_sizes)[mask])) if mask.any() else np.nan

        return ConformalPredictionResult(
            prediction_sets=prediction_sets,
            set_sizes=set_sizes,
            sample_thresholds=thresholds,
            thresholds=dict(self._thresholds),
            coverage=coverage,
            per_bin_coverage=per_bin_coverage,
            per_bin_set_size=per_bin_set_size,
        )


__all__ = ["MondrianConformalClassifier", "ConformalPredictionResult"]
