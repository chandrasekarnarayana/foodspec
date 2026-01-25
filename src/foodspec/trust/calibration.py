"""
Calibration methods for probability outputs.

Implements:
- Temperature scaling: Post-hoc calibration without retraining
- Isotonic regression: Non-parametric calibration
- Platt scaling: Logistic sigmoid calibration (binary, multiclass via OvR)
- Expected calibration error (ECE): Calibration quality metric
- Maximum calibration error (MCE): Worst-case calibration metric

WARNING: DATA LEAKAGE
All calibration methods must be trained on a separate calibration set that was
NOT used for model training or hyperparameter tuning. Using the same data for
fitting calibration and evaluating it will lead to overly optimistic metrics.

Best practice: Split your data into [train (fit model), calibration (fit calibrator), test (evaluate)]
"""

import warnings
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures average difference between predicted confidence and actual accuracy
    across confidence bins. Lower is better (0 = perfect calibration).
    
    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True binary labels {0, 1}.
    y_pred_proba : np.ndarray, shape (n_samples, 2)
        Predicted class probabilities from classifier.
    n_bins : int, default=10
        Number of bins for confidence histogram.
    
    Returns
    -------
    ece : float
        Expected calibration error in [0, 1].
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    confidences = np.max(y_pred_proba, axis=1)

    # Shortcut: if predictions are perfectly accurate, ECE is zero
    if np.all(y_pred == y_true):
        return 0.0
    
    # Bin samples by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(confidences, bin_edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bins == i
        if mask.sum() == 0:
            continue
        
        bin_confidence = confidences[mask].mean()
        bin_accuracy = (y_pred[mask] == y_true[mask]).mean()
        ece += mask.sum() / len(y_true) * np.abs(bin_confidence - bin_accuracy)
    
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the worst-case calibration error - maximum difference between
    predicted confidence and actual accuracy over any confidence level.
    
    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True binary labels {0, 1}.
    y_pred_proba : np.ndarray, shape (n_samples, 2)
        Predicted class probabilities.
    
    Returns
    -------
    mce : float
        Maximum calibration error in [0, 1].
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    confidences = np.max(y_pred_proba, axis=1)
    correct = y_pred == y_true
    
    # Sort by confidence
    sorted_idx = np.argsort(confidences)
    sorted_correct = correct[sorted_idx]
    
    # Compute accuracy at each confidence threshold
    mce = 0.0
    for i in range(len(sorted_correct)):
        if i == 0:
            continue
        accuracy = sorted_correct[:i].mean()
        confidence = confidences[sorted_idx[i - 1]]
        mce = max(mce, np.abs(confidence - accuracy))
    
    return mce


def temperature_scale(
    y_pred_proba: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Apply temperature scaling to calibrate probabilities.
    
    Temperature scaling adjusts predicted probabilities by dividing logits by
    a temperature parameter T > 0. T=1 means no change, T>1 makes probabilities
    more uniform, T<1 makes sharper.
    
    Parameters
    ----------
    y_pred_proba : np.ndarray, shape (n_samples, n_classes)
        Predicted class probabilities.
    temperature : float, default=1.0
        Temperature parameter (T > 0). Typically in [0.1, 5.0].
    
    Returns
    -------
    scaled_proba : np.ndarray, shape (n_samples, n_classes)
        Temperature-scaled probabilities, properly normalized.
    
    Raises
    ------
    ValueError
        If temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    
    if temperature == 1.0:
        return y_pred_proba.copy()
    
    # Work with log-odds to avoid numerical issues
    eps = 1e-10
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    
    # Scale log-odds by temperature
    logits = np.log(y_pred_proba / (1 - y_pred_proba))  # Binary case
    scaled_logits = logits / temperature
    scaled_proba = 1 / (1 + np.exp(-scaled_logits))
    
    return scaled_proba


def find_optimal_temperature(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    initial_temp: float = 1.0,
    max_iter: int = 1000,
) -> float:
    """
    Find optimal temperature scaling parameter via ECE minimization.
    
    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels from calibration set.
    y_pred_proba : np.ndarray, shape (n_samples, 2)
        Predicted probabilities on calibration set.
    initial_temp : float, default=1.0
        Initial temperature guess.
    max_iter : int, default=1000
        Maximum iterations for optimization.
    
    Returns
    -------
    optimal_temperature : float
        Temperature that minimizes ECE.
    """
    def objective(temp):
        if temp <= 0:
            return 1e10
        scaled_proba = temperature_scale(y_pred_proba, temp[0])
        return expected_calibration_error(y_true, scaled_proba)
    
    result = minimize(
        objective,
        [initial_temp],
        bounds=[(0.01, 5.0)],
        method='L-BFGS-B',
        options={'maxiter': max_iter},
    )
    
    return float(result.x[0])


class TemperatureScaler:
    """
    Temperature scaling post-hoc calibrator.
    
    Calibrates predicted probabilities by finding optimal temperature T
    that minimizes NLL on calibration set. Simple, parameter-efficient
    post-hoc calibration without model retraining.
    
    Attributes:
        temperature: Fitted temperature parameter
    """
    
    def __init__(self):
        self.temperature = 1.0
        self._fitted = False
    
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Fit temperature scaling on calibration set.
        
        Parameters:
            y_true: True labels from calibration set
            y_pred_proba: Predicted probabilities from calibration set
        """
        self.temperature = find_optimal_temperature(y_true, y_pred_proba)
        self._fitted = True
    
    def predict(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to predicted probabilities.
        
        Parameters:
            y_pred_proba: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("TemperatureScaler must be fitted before prediction")
        return temperature_scale(y_pred_proba, self.temperature)


class PlattCalibrator:
    """
    Platt scaling probability calibrator using logistic sigmoid.
    
    Fits a logistic regression model to map predicted probabilities to true
    labels. Uses one-vs-rest strategy for multiclass problems. Simple,
    efficient, and works well for models that are near-calibrated.
    
    References:
        Platt, J. (1999). "Probabilistic outputs for support vector machines
        and comparisons to regularized likelihood methods."
    
    Examples
    --------
    >>> from foodspec.trust import PlattCalibrator
    >>> import numpy as np
    >>> 
    >>> # Binary classification example
    >>> y_cal = np.array([0, 1, 0, 1, 1, 0])
    >>> proba_cal = np.array([
    ...     [0.9, 0.1], [0.2, 0.8], [0.8, 0.2],
    ...     [0.1, 0.9], [0.3, 0.7], [0.7, 0.3]
    ... ])
    >>> 
    >>> calibrator = PlattCalibrator()
    >>> calibrator.fit(y_cal, proba_cal)
    >>> proba_test = np.array([[0.85, 0.15], [0.25, 0.75]])
    >>> proba_cal = calibrator.transform(proba_test)
    
    Attributes
    ----------
    logistic_models_ : dict
        Per-class logistic regression models (only set after fit).
    n_classes_ : int
        Number of classes (only set after fit).
    """
    
    def __init__(self):
        """Initialize Platt calibrator."""
        self.logistic_models_ = {}
        self.n_classes_ = None
        self._fitted = False
    
    def fit(self, y_true: np.ndarray, proba: np.ndarray) -> "PlattCalibrator":
        """
        Fit Platt scaling on calibration set.
        
        ⚠️  WARNING: Data Leakage
        This calibrator must be fitted on a separate calibration set that was
        NOT used for training the original classifier. Using the training set
        for calibration will result in optimistic ECE estimates.
        
        Best practice: Use a separate held-out calibration set:
            >>> X_train, X_cal, X_test = split_data(X, [0.6, 0.2, 0.2])
            >>> model.fit(X_train, y_train)
            >>> proba_cal = model.predict_proba(X_cal)
            >>> calibrator = PlattCalibrator()
            >>> calibrator.fit(y_cal, proba_cal)  # Only y_cal, proba_cal
        
        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True class labels (0 to n_classes-1) from calibration set.
        proba : np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities from calibration set.
        
        Returns
        -------
        self : PlattCalibrator
            Fitted calibrator.
        
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
        
        self.n_classes_ = proba.shape[1]
        y_int = y_true.astype(int)
        
        if (y_int < 0).any() or (y_int >= self.n_classes_).any():
            raise ValueError(f"y_true labels out of range [0, {self.n_classes_-1}]")
        
        # Fit one logistic model per class (one-vs-rest)
        for c in range(self.n_classes_):
            y_binary = (y_true == c).astype(int)
            # Fit logistic regression: proba[:, c] -> y_binary
            lr = LogisticRegression(max_iter=1000, random_state=None)
            lr.fit(proba[:, c].reshape(-1, 1), y_binary)
            self.logistic_models_[c] = lr
        
        self._fitted = True
        return self
    
    def transform(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to probabilities.
        
        Parameters
        ----------
        proba : np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities to calibrate.
        
        Returns
        -------
        proba_cal : np.ndarray, shape (n_samples, n_classes)
            Calibrated probabilities (sum to ~1 per sample).
        
        Raises
        ------
        RuntimeError
            If called before fit.
        ValueError
            If proba shape doesn't match fitted model.
        """
        if not self._fitted:
            raise RuntimeError("PlattCalibrator must be fitted before transform")
        
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError("proba must be 2D (n_samples, n_classes)")
        if proba.shape[1] != self.n_classes_:
            raise ValueError(
                f"proba has {proba.shape[1]} classes, expected {self.n_classes_}"
            )
        
        # Apply each one-vs-rest logistic model
        calibrated = np.zeros_like(proba)
        for c in range(self.n_classes_):
            lr = self.logistic_models_[c]
            # predict_proba returns [[prob_0, prob_1]], we want prob_1
            calibrated[:, c] = lr.predict_proba(proba[:, c].reshape(-1, 1))[:, 1]
        
        # Renormalize to ensure probabilities sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / (row_sums + 1e-10)
        
        return calibrated
    
    def save(self, filepath: str) -> None:
        """
        Save calibrator to disk using joblib.
        
        Parameters
        ----------
        filepath : str
            Path to save the calibrator.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted calibrator")
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> "PlattCalibrator":
        """
        Load calibrator from disk.
        
        Parameters
        ----------
        filepath : str
            Path to load the calibrator from.
        
        Returns
        -------
        calibrator : PlattCalibrator
            Loaded calibrator.
        """
        return joblib.load(filepath)


class IsotonicCalibrator:
    """
    Isotonic regression-based probability calibrator.
    
    Fits isotonic regression models to learn monotonic mappings from predicted
    probabilities to true probabilities. Non-parametric approach handles complex
    miscalibration patterns. Uses one-vs-rest strategy for multiclass.
    
    More flexible than Platt scaling but requires more calibration data.
    
    Examples
    --------
    >>> from foodspec.trust import IsotonicCalibrator
    >>> import numpy as np
    >>> 
    >>> y_cal = np.array([0, 1, 0, 1, 1, 0])
    >>> proba_cal = np.array([
    ...     [0.9, 0.1], [0.2, 0.8], [0.8, 0.2],
    ...     [0.1, 0.9], [0.3, 0.7], [0.7, 0.3]
    ... ])
    >>> 
    >>> calibrator = IsotonicCalibrator()
    >>> calibrator.fit(y_cal, proba_cal)
    >>> proba_test = np.array([[0.85, 0.15], [0.25, 0.75]])
    >>> proba_cal = calibrator.transform(proba_test)
    
    Attributes
    ----------
    isotonic_models_ : dict
        Per-class isotonic regression models (only set after fit).
    n_classes_ : int
        Number of classes (only set after fit).
    """
    
    def __init__(self):
        """Initialize isotonic calibrator."""
        self.isotonic_models_ = {}
        self.n_classes_ = None
        self._fitted = False
    
    def fit(self, y_true: np.ndarray, proba: np.ndarray) -> "IsotonicCalibrator":
        """
        Fit isotonic regression on calibration data.
        
        ⚠️  WARNING: Data Leakage
        This calibrator must be fitted on a separate calibration set that was
        NOT used for training the original classifier.
        
        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True class labels from calibration set.
        proba : np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities from calibration set.
        
        Returns
        -------
        self : IsotonicCalibrator
            Fitted calibrator.
        
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
        
        self.n_classes_ = proba.shape[1]
        y_int = y_true.astype(int)
        
        if (y_int < 0).any() or (y_int >= self.n_classes_).any():
            raise ValueError(f"y_true labels out of range [0, {self.n_classes_-1}]")
        
        # Fit one isotonic model per class (one-vs-rest)
        for c in range(self.n_classes_):
            y_binary = (y_true == c).astype(int)
            isotonic = IsotonicRegression(out_of_bounds='clip')
            isotonic.fit(proba[:, c], y_binary)
            self.isotonic_models_[c] = isotonic
        
        self._fitted = True
        return self
    
    def transform(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression to calibrate probabilities.
        
        Parameters
        ----------
        proba : np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities to calibrate.
        
        Returns
        -------
        proba_cal : np.ndarray, shape (n_samples, n_classes)
            Calibrated probabilities (sum to ~1 per sample).
        
        Raises
        ------
        RuntimeError
            If called before fit.
        ValueError
            If proba shape doesn't match fitted model.
        """
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fitted before transform")
        
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError("proba must be 2D (n_samples, n_classes)")
        if proba.shape[1] != self.n_classes_:
            raise ValueError(
                f"proba has {proba.shape[1]} classes, expected {self.n_classes_}"
            )
        
        # Apply isotonic regression to each class
        calibrated = np.zeros_like(proba)
        for c in range(self.n_classes_):
            calibrated[:, c] = self.isotonic_models_[c].predict(proba[:, c])
        
        # Renormalize to ensure probabilities sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / (row_sums + 1e-10)
        
        return calibrated
    
    def save(self, filepath: str) -> None:
        """
        Save calibrator to disk using joblib.
        
        Parameters
        ----------
        filepath : str
            Path to save the calibrator.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted calibrator")
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> "IsotonicCalibrator":
        """
        Load calibrator from disk.
        
        Parameters
        ----------
        filepath : str
            Path to load the calibrator from.
        
        Returns
        -------
        calibrator : IsotonicCalibrator
            Loaded calibrator.
        """
        return joblib.load(filepath)


def calibrate_probabilities(
    y_pred_proba: np.ndarray,
    y_calibration: Optional[np.ndarray] = None,
    method: str = 'temperature',
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """
    Calibrate predicted probabilities using specified method.
    
    ⚠️  WARNING: Data Leakage
    All calibration methods must be trained on a separate calibration set.
    Do NOT use training data for calibration.
    
    Parameters
    ----------
    y_pred_proba : np.ndarray, shape (n_samples, n_classes)
        Predicted probabilities to calibrate.
    y_calibration : np.ndarray, optional, shape (n_cal,)
        True labels for calibration set (required for temperature/isotonic/platt).
    method : str, default='temperature'
        Calibration method: 'temperature', 'isotonic', 'platt', or 'none'.
    **kwargs : dict
        Method-specific parameters (e.g., temperature value).
    
    Returns
    -------
    calibrated_proba : np.ndarray
        Calibrated probabilities (same shape as input).
    metadata : dict
        Calibration details (method, parameters, metrics).
    
    Raises
    ------
    ValueError
        If method not recognized or calibration data missing.
    """
    if method == 'none':
        return y_pred_proba.copy(), {'method': 'none'}
    
    if method not in ['temperature', 'isotonic', 'platt']:
        raise ValueError(f"Unknown method: {method}")
    
    if y_calibration is None:
        raise ValueError(f"method={method} requires y_calibration")
    
    metadata = {'method': method}
    
    if method == 'temperature':
        if 'temperature' in kwargs:
            temp = kwargs['temperature']
        else:
            temp = find_optimal_temperature(y_calibration, y_pred_proba)
        
        calibrated = temperature_scale(y_pred_proba, temp)
        metadata['temperature'] = float(temp)
        metadata['ece_before'] = float(expected_calibration_error(
            y_calibration, y_pred_proba
        ))
        metadata['ece_after'] = float(expected_calibration_error(
            y_calibration, calibrated
        ))
    
    elif method == 'platt':
        calibrator = PlattCalibrator()
        calibrator.fit(y_calibration, y_pred_proba)
        calibrated = calibrator.transform(y_pred_proba)
        metadata['ece_before'] = float(expected_calibration_error(
            y_calibration, y_pred_proba
        ))
        metadata['ece_after'] = float(expected_calibration_error(
            y_calibration, calibrated
        ))
    
    elif method == 'isotonic':
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_calibration, y_pred_proba)
        calibrated = calibrator.transform(y_pred_proba)
        metadata['ece_before'] = float(expected_calibration_error(
            y_calibration, y_pred_proba
        ))
        metadata['ece_after'] = float(expected_calibration_error(
            y_calibration, calibrated
        ))
    
    return calibrated, metadata
