"""
Mondrian conformal prediction for multiclass classification with bin conditioning.

The implementation follows the split-conformal recipe: fit on train, compute
nonconformity scores on a held-out calibration split, and build prediction sets
that achieve target coverage. Conditioning by metadata bins (e.g., stage or
instrument) yields a Mondrian variant with per-bin thresholds and conditional
coverage guarantees.

Key References:
    Barber et al. (2019): "Predictive inference with the jackknife"
    Lei et al. (2018): "Classification with honest confidence"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _to_bin_key(x: object) -> str:
    """Convert bin value to string key for threshold lookup."""
    if x is None:
        return "__global__"
    return str(x)


def _mondrian_quantile(scores: np.ndarray, target_coverage: float) -> float:
    """Compute quantile threshold for target coverage.

    Formula: threshold = quantile(ceil((n + 1) * target_coverage) / n)
    where n = number of scores.

    This ensures coverage ≥ target_coverage with finite samples.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("scores cannot be empty")

    # Quantile level: ensures coverage guarantee
    quantile_level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
    threshold = float(np.quantile(scores, quantile_level, method='lower'))

    return threshold


@dataclass
class ConformalPredictionResult:
    """Conformal prediction results with coverage information.

    Attributes
    ----------
    prediction_sets : list of list of int
        Predicted class sets (list of class indices per sample).
    set_sizes : list of int
        Number of classes in each prediction set.
    sample_thresholds : list of float
        Threshold used for each sample's prediction set.
    thresholds : dict
        All thresholds (global + per-bin).
    coverage : float or None
        Observed marginal coverage if y_true provided, else None.
    per_bin_coverage : dict
        Per-bin empirical coverage (key -> coverage value).
    per_bin_set_size : dict
        Per-bin mean set size.
    """
    prediction_sets: List[List[int]]
    set_sizes: List[int]
    sample_thresholds: List[float]
    thresholds: Dict[str, float]
    coverage: Optional[float] = None
    per_bin_coverage: Optional[Dict[str, float]] = None
    per_bin_set_size: Optional[Dict[str, float]] = None

    def to_dataframe(
        self,
        y_true: Optional[np.ndarray] = None,
        bin_values: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Convert results to DataFrame.

        Parameters
        ----------
        y_true : ndarray, optional
            True labels (optional). If provided, adds 'covered' column.
        bin_values : ndarray, optional
            Bin identifiers (optional). If provided, adds 'bin' column.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: set_size, (covered), (bin), set_members
        """
        n = len(self.set_sizes)
        data = {
            'set_size': self.set_sizes,
            'threshold': self.sample_thresholds,
            'set_members': [str(s) for s in self.prediction_sets],
        }

        if y_true is not None:
            y_true = np.asarray(y_true)
            data['covered'] = [int(y_true[i] in self.prediction_sets[i]) for i in range(n)]

        if bin_values is not None:
            bin_values = np.asarray(bin_values)
            data['bin'] = bin_values

        return pd.DataFrame(data)


class MondrianConformalClassifier:
    """Split-conformal predictor for multiclass classification.

    Implements Mondrian conformal prediction for probability-based classifiers.
    Computes nonconformity scores as 1 - p_true on calibration data, then
    builds prediction sets (via conformal prediction) with optional per-bin
    conditioning for stratified coverage.

    Parameters
    ----------
    alpha : float, default 0.1
        Significance level for coverage. Achieves coverage ≥ 1 - alpha.
        E.g., alpha=0.1 → target coverage = 0.9.
    condition_key : str, optional
        If provided, compute per-bin thresholds using this column from meta_cal.
        Enables stratified/conditional coverage guarantees.
    min_bin_size : int, default 20
        Minimum samples per bin to compute separate threshold. Falls back to
        global threshold if bin smaller than this.

    Attributes
    ----------
    _thresholds : dict
        Global + per-bin nonconformity thresholds.
    _fitted : bool
        Whether calibration data has been processed.

    Examples
    --------
    >>> import numpy as np
    >>> from foodspec.trust import MondrianConformalClassifier
    >>>
    >>> # Create synthetic multiclass data
    >>> np.random.seed(42)
    >>> y_cal = np.random.randint(0, 3, 100)
    >>> proba_cal = np.random.dirichlet([1, 1, 1], size=100)
    >>>
    >>> # Instantiate and fit
    >>> cp = MondrianConformalClassifier(alpha=0.1)
    >>> cp.fit(y_cal, proba_cal)
    >>>
    >>> # Generate prediction sets
    >>> proba_test = np.random.dirichlet([1, 1, 1], size=50)
    >>> result = cp.predict_sets(proba_test)
    >>> print(f"Average set size: {np.mean(result.set_sizes):.2f}")

    >>> # With conditional coverage
    >>> meta_cal = np.array(['batch_A'] * 50 + ['batch_B'] * 50)
    >>> cp_cond = MondrianConformalClassifier(alpha=0.1, condition_key='batch')
    >>> cp_cond.fit(y_cal, proba_cal, meta_cal=meta_cal)
    >>> result_cond = cp_cond.predict_sets(proba_test, meta_test=meta_test)
    >>> print(result_cond.per_bin_coverage)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        condition_key: Optional[str] = None,
        min_bin_size: int = 20,
    ):
        """Initialize Mondrian conformal classifier."""
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if min_bin_size < 1:
            raise ValueError(f"min_bin_size must be ≥ 1, got {min_bin_size}")

        self.alpha = alpha
        self.condition_key = condition_key
        self.min_bin_size = min_bin_size

        self._thresholds: Dict[str, float] = {}
        self._fitted = False
        self._n_classes: Optional[int] = None

    def fit(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        meta_cal: Optional[np.ndarray] = None,
    ) -> "MondrianConformalClassifier":
        """
        Fit conformal predictor on calibration data.

        Computes nonconformity scores (1 - p_true) and optional per-bin thresholds.

        ⚠️ IMPORTANT: Call this only on a separate calibration set that was NOT
        used for training the base classifier. Using training data will lead to
        overly optimistic coverage estimates (data leakage).

        Best practice: Split data into [train (fit base model), cal (fit CP), test]

        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True class labels (0 to n_classes - 1).
        proba : np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities from base classifier.
        meta_cal : np.ndarray, optional, shape (n_samples,)
            Metadata values for binning (e.g., batch IDs, stage names).
            If condition_key is set, used to compute per-bin thresholds.

        Returns
        -------
        self : MondrianConformalClassifier
            Fitted classifier.

        Raises
        ------
        ValueError
            If shapes mismatch or labels out of range.
        """
        y_true = np.asarray(y_true)
        proba = np.asarray(proba, dtype=float)

        if y_true.ndim != 1:
            raise ValueError("y_true must be 1D")
        if proba.ndim != 2:
            raise ValueError("proba must be 2D (n_samples, n_classes)")
        if y_true.shape[0] != proba.shape[0]:
            raise ValueError("y_true and proba must have same number of samples")

        self._n_classes = proba.shape[1]
        y_int = y_true.astype(int)

        if (y_int < 0).any() or (y_int >= self._n_classes).any():
            raise ValueError(f"y_true labels out of range [0, {self._n_classes - 1}]")

        # Nonconformity score: 1 - p_true
        scores = 1.0 - proba[np.arange(proba.shape[0]), y_int]

        # Global threshold
        threshold_global = _mondrian_quantile(scores, 1.0 - self.alpha)
        self._thresholds["__global__"] = threshold_global
        self._thresholds["global"] = threshold_global

        # Per-bin thresholds if conditioning enabled
        if self.condition_key is not None and meta_cal is not None:
            meta_cal = np.asarray(meta_cal)
            if meta_cal.shape[0] != proba.shape[0]:
                raise ValueError("meta_cal length must match calibration data")

            bin_keys = [_to_bin_key(m) for m in meta_cal]
            unique_bins = np.unique(bin_keys)

            for bin_key in unique_bins:
                mask = np.array(bin_keys) == bin_key
                bin_scores = scores[mask]

                # Only compute separate threshold if bin is large enough
                if len(bin_scores) >= self.min_bin_size:
                    self._thresholds[bin_key] = _mondrian_quantile(
                        bin_scores, 1.0 - self.alpha
                    )
                # else: use global threshold (fallback)

        self._fitted = True
        return self

    def predict_sets(
        self,
        proba: np.ndarray,
        meta_test: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
    ) -> ConformalPredictionResult:
        """
        Generate prediction sets for test samples.

        Returns sets of class indices where predicted probability ≥ 1 - threshold.

        Parameters
        ----------
        proba : np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities from base classifier.
        meta_test : np.ndarray, optional, shape (n_samples,)
            Metadata for test samples (used for per-bin threshold lookup).
        y_true : np.ndarray, optional, shape (n_samples,)
            True labels (optional). If provided, computes empirical coverage.

        Returns
        -------
        result : ConformalPredictionResult
            Prediction sets, set sizes, thresholds, and optional coverage metrics.

        Raises
        ------
        RuntimeError
            If called before fit().
        ValueError
            If shapes mismatch.
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted; call fit() first")

        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError("proba must be 2D (n_samples, n_classes)")
        if proba.shape[1] != self._n_classes:
            raise ValueError(
                f"proba has {proba.shape[1]} classes, expected {self._n_classes}"
            )

        n_samples = proba.shape[0]

        # Determine thresholds for each sample
        if meta_test is not None:
            meta_test = np.asarray(meta_test)
            if meta_test.shape[0] != n_samples:
                raise ValueError("meta_test length must match proba")
            bin_keys = [_to_bin_key(m) for m in meta_test]
        else:
            bin_keys = ["__global__"] * n_samples

        # Build prediction sets
        thresholds: List[float] = []
        prediction_sets: List[List[int]] = []

        for i in range(n_samples):
            bin_key = bin_keys[i]
            # Lookup threshold: try bin-specific first, fallback to global
            threshold = self._thresholds.get(
                bin_key,
                self._thresholds.get("__global__")
            )

            if threshold is None:
                raise RuntimeError("No threshold available for prediction")

            thresholds.append(float(threshold))

            # Prediction set: all classes with p ≥ 1 - threshold
            # Numerically robust check: p ≥ 1 - t is equivalent to p - (1 - t) ≥ 0
            keep = [
                int(c)
                for c in range(self._n_classes)
                if proba[i, c] >= 1.0 - threshold - 1e-12
            ]

            # Fallback: include top-1 class if set empty
            if not keep:
                keep = [int(np.argmax(proba[i]))]

            prediction_sets.append(keep)

        # Compute coverage metrics if labels provided
        coverage: Optional[float] = None
        per_bin_coverage: Dict[str, float] = {}
        per_bin_set_size: Dict[str, float] = {}

        if y_true is not None:
            y_true = np.asarray(y_true)
            if y_true.shape[0] != n_samples:
                raise ValueError("y_true length must match predictions")

            # Marginal coverage
            covered = np.array([
                int(y_true[i] in prediction_sets[i])
                for i in range(n_samples)
            ], dtype=float)
            coverage = float(np.mean(covered))

            # Per-bin coverage and set size
            unique_bins = np.unique(bin_keys)
            for bin_key in unique_bins:
                mask = np.array(bin_keys) == bin_key
                if mask.any():
                    bin_covered = covered[mask]
                    bin_sizes = np.array([
                        len(prediction_sets[i]) for i in np.where(mask)[0]
                    ])
                    per_bin_coverage[bin_key] = float(np.mean(bin_covered))
                    per_bin_set_size[bin_key] = float(np.mean(bin_sizes))

        return ConformalPredictionResult(
            prediction_sets=prediction_sets,
            set_sizes=[len(s) for s in prediction_sets],
            sample_thresholds=thresholds,
            thresholds=dict(self._thresholds),
            coverage=coverage,
            per_bin_coverage=per_bin_coverage if per_bin_coverage else None,
            per_bin_set_size=per_bin_set_size if per_bin_set_size else None,
        )

    def coverage_report(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        meta_test: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive coverage report.

        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True labels.
        proba : np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities.
        meta_test : np.ndarray, optional
            Metadata for per-bin breakdown.

        Returns
        -------
        df : pd.DataFrame
            Report with columns:
            - bin (str): bin identifier
            - n_samples (int): number of samples in bin
            - coverage (float): empirical coverage in bin
            - target_coverage (float): 1 - alpha
            - avg_set_size (float): mean set size
            - threshold (float): threshold used
        """
        result = self.predict_sets(proba, meta_test=meta_test, y_true=y_true)

        if result.coverage is None:
            raise ValueError("No coverage computed (y_true not provided)")

        # Global row
        rows = [{
            'bin': '__global__',
            'n_samples': len(y_true),
            'coverage': result.coverage,
            'target_coverage': 1.0 - self.alpha,
            'avg_set_size': float(np.mean(result.set_sizes)),
            'threshold': self._thresholds.get('__global__', self._thresholds.get('global')),
        }]

        # Per-bin rows if available
        if result.per_bin_coverage:
            if meta_test is not None:
                meta_test = np.asarray(meta_test)
                bin_keys = [_to_bin_key(m) for m in meta_test]
                unique_bins = np.unique(bin_keys)

                for bin_key in unique_bins:
                    if bin_key == "__global__":
                        continue
                    mask = np.array(bin_keys) == bin_key
                    rows.append({
                        'bin': bin_key,
                        'n_samples': int(mask.sum()),
                        'coverage': result.per_bin_coverage.get(bin_key, np.nan),
                        'target_coverage': 1.0 - self.alpha,
                        'avg_set_size': result.per_bin_set_size.get(bin_key, np.nan),
                        'threshold': self._thresholds.get(bin_key, self._thresholds.get('__global__')),
                    })

        return pd.DataFrame(rows)


class ConformalPredictor(MondrianConformalClassifier):
    """Backward-compatible alias for MondrianConformalClassifier."""

__all__ = ["MondrianConformalClassifier", "ConformalPredictionResult", "ConformalPredictor"]
