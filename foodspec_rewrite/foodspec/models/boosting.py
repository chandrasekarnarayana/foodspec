"""
Gradient boosting classifier wrappers with optional dependencies.

This module provides wrappers for XGBoost and LightGBM with lazy imports.
If the optional dependencies are not installed, clear error messages guide installation.

Classes
-------
XGBoostClassifierWrapper
    Wrapper for XGBoost classifier (optional dependency).
LightGBMClassifierWrapper
    Wrapper for LightGBM classifier (optional dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np

from foodspec.models.base import BaseEstimator


@dataclass
class XGBoostClassifierWrapper(BaseEstimator):
    """Wrapper over XGBoost classifier with spectroscopy defaults.

    XGBoost is an optional dependency. Install with: pip install xgboost

    Fast gradient boosting for classification with excellent performance on
    high-dimensional data. Provides feature importance and probability estimates.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of boosting rounds. More rounds = better but slower.
        Suitable for spectroscopy: 100-200 works well.

    max_depth : int, default 6
        Maximum tree depth. Controls model complexity.
        Suitable for spectroscopy: 3-8 prevents overfitting.

    learning_rate : float, default 0.1
        Step size shrinkage to prevent overfitting. Range: 0.01-0.3.

    subsample : float, default 0.8
        Fraction of samples used per tree. Range: 0.5-1.0.

    colsample_bytree : float, default 0.8
        Fraction of features used per tree. Reduces correlation.

    reg_alpha : float, default 0.0
        L1 regularization. Higher = more regularization.

    reg_lambda : float, default 1.0
        L2 regularization. Higher = more regularization.

    scale_pos_weight : Optional[float], default None
        Balances positive/negative weights for imbalanced data.
        Set to (n_negative / n_positive) for class balance.

    random_state : int, default 0
        Random seed for reproducibility.

    n_jobs : int, default -1
        Number of parallel threads. -1 uses all available cores.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 500)  # Spectroscopy: many features
    >>> y = np.array([0, 1] * 50)
    >>>
    >>> # Default setup
    >>> clf = XGBoostClassifierWrapper()
    >>> clf.fit(X, y)
    XGBoostClassifierWrapper(...)
    >>>
    >>> # Predictions with probabilities
    >>> proba = clf.predict_proba(X)
    >>> proba.shape
    (100, 2)
    >>>
    >>> # Feature importance
    >>> importance = clf.get_feature_importance()
    >>> importance.shape
    (500,)

    Notes
    -----
    XGBoost must be installed separately: pip install xgboost
    If not installed, a clear error message with installation instructions is raised.
    """

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: Optional[float] = None
    random_state: int = 0
    n_jobs: int = -1

    _model: Any = field(default=None, init=False, repr=False)
    _xgboost_available: bool = field(default=False, init=False, repr=False)
    _xgb_module: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize and validate parameters."""
        self._check_xgboost_availability()
        self._validate_params()

    def _check_xgboost_availability(self) -> None:
        """Check if XGBoost is available and import it."""
        try:
            import xgboost as xgb

            self._xgb_module = xgb
            self._xgboost_available = True
        except ImportError as e:
            raise ImportError(
                "XGBoost is not installed. This is an optional dependency.\n"
                "To use XGBoostClassifierWrapper, install it with:\n"
                "  pip install xgboost\n"
                "or install foodspec with the 'boosting' extra:\n"
                "  pip install foodspec[boosting]\n"
                f"Original error: {e}"
            ) from e

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")

        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")

        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")

        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")

        if self.reg_alpha < 0.0:
            raise ValueError("reg_alpha must be non-negative")

        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be non-negative")

        if self.scale_pos_weight is not None and self.scale_pos_weight <= 0.0:
            raise ValueError("scale_pos_weight must be positive")

    def _make_model(self) -> Any:
        """Create XGBoost classifier."""
        return self._xgb_module.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostClassifierWrapper:
        """Fit XGBoost classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        self._model = self._make_model()
        self._model.fit(X, y, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self._model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Feature importance scores (normalized to sum to 1).

        Raises
        ------
        RuntimeError
            If model is not fitted.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = self._model.feature_importances_
        # Normalize to sum to 1
        return importance / importance.sum()

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : Path | str
            Path to save model.
        """
        if self._model is None:
            raise RuntimeError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the entire wrapper (including XGBoost model)
        joblib.dump(
            {
                "model": self._model,
                "params": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "subsample": self.subsample,
                    "colsample_bytree": self.colsample_bytree,
                    "reg_alpha": self.reg_alpha,
                    "reg_lambda": self.reg_lambda,
                    "scale_pos_weight": self.scale_pos_weight,
                    "random_state": self.random_state,
                    "n_jobs": self.n_jobs,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> XGBoostClassifierWrapper:
        """Load model from disk.

        Parameters
        ----------
        path : Path | str
            Path to load model from.

        Returns
        -------
        XGBoostClassifierWrapper
            Loaded model.
        """
        data = joblib.load(path)
        wrapper = cls(**data["params"])
        wrapper._model = data["model"]
        return wrapper

    @classmethod
    def default_hyperparams(cls) -> Dict[str, Any]:
        """Return spectroscopy-optimized default hyperparameters.

        Returns
        -------
        dict
            Dictionary of default hyperparameters.
        """
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": None,
            "random_state": 0,
            "n_jobs": -1,
        }

    @classmethod
    def fast(cls, **kwargs: Any) -> XGBoostClassifierWrapper:
        """Create classifier for quick iteration (fewer rounds).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        XGBoostClassifierWrapper
            Fast classifier with n_estimators=50.
        """
        params = cls.default_hyperparams()
        params["n_estimators"] = 50
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def strong_regularization(cls, **kwargs: Any) -> XGBoostClassifierWrapper:
        """Create classifier with strong regularization.

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        XGBoostClassifierWrapper
            Strongly regularized classifier.
        """
        params = cls.default_hyperparams()
        params["max_depth"] = 3
        params["reg_alpha"] = 1.0
        params["reg_lambda"] = 10.0
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def weak_regularization(cls, **kwargs: Any) -> XGBoostClassifierWrapper:
        """Create classifier with weak regularization.

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        XGBoostClassifierWrapper
            Weakly regularized classifier.
        """
        params = cls.default_hyperparams()
        params["max_depth"] = 10
        params["reg_lambda"] = 0.1
        params.update(kwargs)
        return cls(**params)


@dataclass
class LightGBMClassifierWrapper(BaseEstimator):
    """Wrapper over LightGBM classifier with spectroscopy defaults.

    LightGBM is an optional dependency. Install with: pip install lightgbm

    Fast gradient boosting optimized for large datasets and high-dimensional data.
    Often faster than XGBoost with similar or better accuracy.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of boosting rounds. More rounds = better but slower.

    max_depth : int, default -1
        Maximum tree depth. -1 means no limit.
        Suitable for spectroscopy: 5-10 prevents overfitting.

    learning_rate : float, default 0.1
        Step size shrinkage to prevent overfitting.

    num_leaves : int, default 31
        Maximum number of leaves per tree. Controls complexity.

    subsample : float, default 0.8
        Fraction of samples used per tree.

    colsample_bytree : float, default 0.8
        Fraction of features used per tree.

    reg_alpha : float, default 0.0
        L1 regularization.

    reg_lambda : float, default 1.0
        L2 regularization.

    class_weight : Optional[str], default None
        Class weight mode. 'balanced' adjusts for imbalanced data.

    random_state : int, default 0
        Random seed for reproducibility.

    n_jobs : int, default -1
        Number of parallel threads. -1 uses all available cores.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 500)
    >>> y = np.array([0, 1] * 50)
    >>>
    >>> clf = LightGBMClassifierWrapper()
    >>> clf.fit(X, y)
    LightGBMClassifierWrapper(...)
    >>>
    >>> proba = clf.predict_proba(X)
    >>> proba.shape
    (100, 2)

    Notes
    -----
    LightGBM must be installed separately: pip install lightgbm
    If not installed, a clear error message with installation instructions is raised.
    """

    n_estimators: int = 100
    max_depth: int = -1
    learning_rate: float = 0.1
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    class_weight: Optional[str] = None
    random_state: int = 0
    n_jobs: int = -1

    _model: Any = field(default=None, init=False, repr=False)
    _lightgbm_available: bool = field(default=False, init=False, repr=False)
    _lgb_module: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize and validate parameters."""
        self._check_lightgbm_availability()
        self._validate_params()

    def _check_lightgbm_availability(self) -> None:
        """Check if LightGBM is available and import it."""
        try:
            import lightgbm as lgb

            self._lgb_module = lgb
            self._lightgbm_available = True
        except ImportError as e:
            raise ImportError(
                "LightGBM is not installed. This is an optional dependency.\n"
                "To use LightGBMClassifierWrapper, install it with:\n"
                "  pip install lightgbm\n"
                "or install foodspec with the 'boosting' extra:\n"
                "  pip install foodspec[boosting]\n"
                f"Original error: {e}"
            ) from e

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")

        if self.max_depth != -1 and self.max_depth <= 0:
            raise ValueError("max_depth must be positive or -1 (no limit)")

        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")

        if self.num_leaves <= 1:
            raise ValueError("num_leaves must be > 1")

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")

        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")

        if self.reg_alpha < 0.0:
            raise ValueError("reg_alpha must be non-negative")

        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be non-negative")

        if self.class_weight is not None and self.class_weight != "balanced":
            raise ValueError("class_weight must be None or 'balanced'")

    def _make_model(self) -> Any:
        """Create LightGBM classifier."""
        return self._lgb_module.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> LightGBMClassifierWrapper:
        """Fit LightGBM classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        self._model = self._make_model()
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self._model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Feature importance scores (normalized to sum to 1).

        Raises
        ------
        RuntimeError
            If model is not fitted.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = self._model.feature_importances_
        # Normalize to sum to 1
        return importance / importance.sum()

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : Path | str
            Path to save model.
        """
        if self._model is None:
            raise RuntimeError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self._model,
                "params": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "num_leaves": self.num_leaves,
                    "subsample": self.subsample,
                    "colsample_bytree": self.colsample_bytree,
                    "reg_alpha": self.reg_alpha,
                    "reg_lambda": self.reg_lambda,
                    "class_weight": self.class_weight,
                    "random_state": self.random_state,
                    "n_jobs": self.n_jobs,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> LightGBMClassifierWrapper:
        """Load model from disk.

        Parameters
        ----------
        path : Path | str
            Path to load model from.

        Returns
        -------
        LightGBMClassifierWrapper
            Loaded model.
        """
        data = joblib.load(path)
        wrapper = cls(**data["params"])
        wrapper._model = data["model"]
        return wrapper

    @classmethod
    def default_hyperparams(cls) -> Dict[str, Any]:
        """Return spectroscopy-optimized default hyperparameters.

        Returns
        -------
        dict
            Dictionary of default hyperparameters.
        """
        return {
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": 0,
            "n_jobs": -1,
        }

    @classmethod
    def fast(cls, **kwargs: Any) -> LightGBMClassifierWrapper:
        """Create classifier for quick iteration (fewer rounds).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LightGBMClassifierWrapper
            Fast classifier with n_estimators=50.
        """
        params = cls.default_hyperparams()
        params["n_estimators"] = 50
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def strong_regularization(cls, **kwargs: Any) -> LightGBMClassifierWrapper:
        """Create classifier with strong regularization.

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LightGBMClassifierWrapper
            Strongly regularized classifier.
        """
        params = cls.default_hyperparams()
        params["max_depth"] = 5
        params["num_leaves"] = 15
        params["reg_alpha"] = 1.0
        params["reg_lambda"] = 10.0
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def weak_regularization(cls, **kwargs: Any) -> LightGBMClassifierWrapper:
        """Create classifier with weak regularization.

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LightGBMClassifierWrapper
            Weakly regularized classifier.
        """
        params = cls.default_hyperparams()
        params["max_depth"] = -1
        params["num_leaves"] = 63
        params["reg_lambda"] = 0.1
        params.update(kwargs)
        return cls(**params)


__all__ = [
    "XGBoostClassifierWrapper",
    "LightGBMClassifierWrapper",
]
