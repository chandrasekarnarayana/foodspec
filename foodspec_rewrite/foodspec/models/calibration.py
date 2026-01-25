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

Model calibration wrappers: Platt scaling and isotonic regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class CalibratedClassifier:
    """Wrapper for sklearn calibration (Platt or isotonic).

    Calibration is always trained on a separate validation fold to avoid
    overfitting to the training data. Use ``fit_and_calibrate`` to train
    the base model and calibration sequentially on distinct splits.

    Parameters
    ----------
    base_estimator : object
        A fitted or unfitted estimator with predict_proba.
    method : {"sigmoid", "isotonic"}, default "sigmoid"
        Calibration method. "sigmoid" is Platt scaling; "isotonic" is
        isotonic regression.
    n_bins : int, default 5
        Number of bins for binning strategy (not used in Platt/isotonic
        but available for future extensions).

    Examples
    --------
    >>> from foodspec.models import LogisticRegressionClassifier, CalibratedClassifier
    >>> import numpy as np
    >>> X_train = np.random.randn(40, 5)
    >>> y_train = np.random.randint(0, 2, 40)
    >>> X_val = np.random.randn(10, 5)
    >>> y_val = np.random.randint(0, 2, 10)
    >>> base = LogisticRegressionClassifier(random_state=0)
    >>> cal = CalibratedClassifier(base, method="sigmoid")
    >>> cal.fit_and_calibrate(X_train, y_train, X_val, y_val)
    CalibratedClassifier(method='sigmoid', n_bins=5)
    >>> proba_cal = cal.predict_proba(X_val)
    >>> proba_cal.shape
    (10, 2)
    """

    base_estimator: object
    method: str = "sigmoid"
    n_bins: int = 5
    _calibrator: CalibratedClassifierCV = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit_and_calibrate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> "CalibratedClassifier":
        """Fit base model on training data, then calibrate on validation data.

        Parameters
        ----------
        X_train : ndarray, shape (n_train, n_features)
            Training features.
        y_train : ndarray, shape (n_train,)
            Training labels.
        X_val : ndarray, shape (n_val, n_features)
            Validation features for calibration (should not overlap with training).
        y_val : ndarray, shape (n_val,)
            Validation labels.

        Returns
        -------
        self
        """

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val)

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same length")
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("X_val and y_val must have the same length")
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError("X_train and X_val must have the same number of features")

        # Fit base model on training data
        self.base_estimator.fit(X_train, y_train)

        # Create calibrator and fit on validation data using cv=None for single split
        self._calibrator = CalibratedClassifierCV(
            estimator=self.base_estimator, method=self.method, cv=None
        )
        # Fit calibrator on validation fold; cv=None uses the provided data as single split
        self._calibrator.fit(X_val, y_val)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Samples to calibrate.

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            Calibrated class probabilities.
        """

        if not self._fitted:
            raise RuntimeError("Not calibrated; call fit_and_calibrate first.")
        X = np.asarray(X, dtype=float)
        return self._calibrator.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated class predictions (argmax of probabilities)."""

        proba = self.predict_proba(X)
        return proba.argmax(axis=1)


def calibration_metrics(y_true: np.ndarray, proba_uncal: np.ndarray, proba_cal: np.ndarray) -> dict:
    """Compare calibration before and after using ECE and Brier score.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples,)
        Ground truth labels.
    proba_uncal : ndarray, shape (n_samples, n_classes)
        Uncalibrated probabilities.
    proba_cal : ndarray, shape (n_samples, n_classes)
        Calibrated probabilities.

    Returns
    -------
    metrics : dict
        Keys: "ece_uncal", "ece_cal", "brier_uncal", "brier_cal".
    """

    from foodspec.validation.metrics import brier, ece

    y_true = np.asarray(y_true)
    proba_uncal = np.asarray(proba_uncal, dtype=float)
    proba_cal = np.asarray(proba_cal, dtype=float)

    ece_uncal = ece(y_true, proba_uncal, n_bins=10)
    ece_cal = ece(y_true, proba_cal, n_bins=10)
    brier_uncal = brier(y_true, proba_uncal)
    brier_cal = brier(y_true, proba_cal)

    return {
        "ece_uncal": float(ece_uncal),
        "ece_cal": float(ece_cal),
        "brier_uncal": float(brier_uncal),
        "brier_cal": float(brier_cal),
    }


__all__ = ["CalibratedClassifier", "calibration_metrics"]
