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

Classical ML model wrappers: Logistic Regression, Linear SVM, SVM with probabilities, and Random Forest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from foodspec.models.base import BaseEstimator


@dataclass
class LogisticRegressionClassifier(BaseEstimator):
    """Wrapper over scikit-learn LogisticRegression with spectroscopy-optimized defaults.

    Provides intelligent solver selection based on penalty, class weight handling,
    and spectroscopy-suitable hyperparameters out of the box.

    Parameters
    ----------
    C : float, default 1.0
        Inverse of regularization strength. Smaller values = stronger regularization.
        Suitable for spectroscopy: values in [0.1, 10.0] work well due to high dimensionality.
    
    penalty : {"l1", "l2", "elasticnet"}, default "l2"
        Regularization type. L2 is default; L1 for sparse solutions.
    
    solver : str, optional
        Optimization algorithm. Auto-selected based on penalty if None:
        - L2: "lbfgs" (Newton-like, good for moderate dims)
        - L1: "liblinear" (only solver supporting L1)
        - elasticnet: "saga" (stochastic, supports elasticnet)
    
    l1_ratio : float, default 0.5
        Mix of L1 and L2 for elasticnet (0=L2 only, 1=L1 only).
        Only used when penalty="elasticnet".
    
    class_weight : {"balanced"} | dict, optional
        Weight adjustment for imbalanced classes.
        "balanced": weight inversely to class frequency.
        dict: explicit weights per class.
    
    max_iter : int, default 1000
        Maximum optimization iterations.
    
    random_state : int, default 0
        Random seed for deterministic training (affects solver randomness).
    
    multi_class : {"auto", "ovr", "multinomial"}, default "auto"
        Strategy for multi-class classification.
        "multinomial" recommended for multi-class problems.
    
    tol : float, default 1e-4
        Convergence tolerance.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 500)  # Spectroscopy: many features
    >>> y = np.array([0, 1] * 50)
    >>> 
    >>> # Default spectroscopy setup
    >>> clf = LogisticRegressionClassifier()
    >>> clf.fit(X, y)
    LogisticRegressionClassifier(...)
    >>> proba = clf.predict_proba(X)
    >>> proba.shape
    (100, 2)
    >>>
    >>> # Spectroscopy defaults
    >>> defaults = LogisticRegressionClassifier.default_hyperparams()
    >>> print(defaults)
    {'C': 1.0, 'penalty': 'l2', 'class_weight': 'balanced', ...}
    >>>
    >>> # Imbalanced dataset
    >>> y_imbalanced = np.array([0] * 80 + [1] * 20)
    >>> clf_balanced = LogisticRegressionClassifier(class_weight="balanced")
    >>> clf_balanced.fit(X, y_imbalanced)
    LogisticRegressionClassifier(...)

    Notes
    -----
    - **Spectroscopy use**: Default C=1.0 works well for high-dimensional data
      (like Raman/IR spectroscopy with 500+ wavelengths).
    - **Class imbalance**: Use class_weight="balanced" when classes are imbalanced.
    - **Sparse solutions**: Use penalty="l1" to identify important features
      (requires solver="liblinear").
    - **Large datasets**: Use penalty="elasticnet" with solver="saga" for
      stochastic gradient descent.
    """

    C: float = 1.0
    penalty: Literal["l1", "l2", "elasticnet"] = "l2"
    solver: Optional[str] = None  # Auto-select if None
    l1_ratio: float = 0.5
    class_weight: Optional[str | Dict[int, float]] = None
    max_iter: int = 1000
    random_state: int = 0
    tol: float = 1e-4

    _model: LogisticRegression = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters after dataclass initialization."""
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate parameter combinations."""
        # Validate penalty
        if self.penalty not in ("l1", "l2", "elasticnet"):
            raise ValueError(
                f"penalty must be 'l1', 'l2', or 'elasticnet'; got '{self.penalty}'"
            )

        # Validate C
        if self.C <= 0:
            raise ValueError(f"C must be positive; got {self.C}")

        # Validate l1_ratio for elasticnet
        if self.penalty == "elasticnet":
            if not 0 <= self.l1_ratio <= 1:
                raise ValueError(
                    f"l1_ratio must be in [0, 1] for elasticnet; got {self.l1_ratio}"
                )

        # Validate class_weight
        if self.class_weight is not None:
            if isinstance(self.class_weight, str) and self.class_weight != "balanced":
                raise ValueError(
                    f"class_weight string must be 'balanced' or None; got '{self.class_weight}'"
                )

        # Validate max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive; got {self.max_iter}")

        # Validate tol
        if self.tol <= 0:
            raise ValueError(f"tol must be positive; got {self.tol}")

    def _get_solver(self) -> str:
        """Select solver based on penalty if not explicitly set.

        Returns
        -------
        str
            Solver name compatible with penalty.
        """
        if self.solver is not None:
            return self.solver

        # Auto-select solver based on penalty
        if self.penalty == "l1":
            return "liblinear"  # Only solver supporting L1
        elif self.penalty == "elasticnet":
            return "saga"  # Supports elasticnet
        else:  # l2
            return "lbfgs"  # Good default for l2

    def _make_model(self) -> LogisticRegression:
        """Create sklearn LogisticRegression with validated parameters.
        
        Returns
        -------
        LogisticRegression
            Configured sklearn model.
        """
        solver = self._get_solver()

        # Build kwargs
        kwargs = {
            "C": self.C,
            "penalty": self.penalty,
            "solver": solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "tol": self.tol,
        }

        # Add l1_ratio only for elasticnet
        if self.penalty == "elasticnet":
            kwargs["l1_ratio"] = self.l1_ratio

        # Add class_weight if specified
        if self.class_weight is not None:
            kwargs["class_weight"] = self.class_weight

        # Suppress sklearn deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'multi_class' was deprecated",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="'solver' was deprecated",
                category=FutureWarning,
            )
            return LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionClassifier:
        """Fit the logistic regression model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray, shape (n_samples,)
            Training labels.

        Returns
        -------
        self
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If X is not 2D or y is not 1D or sizes don't match.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples; got {X.shape[0]} vs {y.shape[0]}"
            )

        self._model = self._make_model()
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted class labels.
        """
        self._ensure_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples, n_classes)
            Class probabilities.
        """
        self._ensure_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return self._model.predict_proba(X)

    def get_coef(self) -> np.ndarray:
        """Get model coefficients (feature weights).

        Returns
        -------
        ndarray, shape (n_classes, n_features) or (n_features,)
            Model coefficients. Shape depends on n_classes and solver.
        """
        self._ensure_fitted()
        return self._model.coef_

    def get_intercept(self) -> np.ndarray:
        """Get model intercepts (biases).

        Returns
        -------
        ndarray, shape (n_classes,) or scalar
            Model intercepts.
        """
        self._ensure_fitted()
        return self._model.intercept_

    @classmethod
    def default_hyperparams(cls) -> Dict[str, Any]:
        """Return spectroscopy-optimized default hyperparameters.

        Returns
        -------
        dict
            Dictionary of default hyperparameters suitable for spectroscopy data.

        Notes
        -----
        These defaults are tuned for typical spectroscopy applications:
        - Moderate regularization (C=1.0) for high-dimensional data
        - L2 penalty for smooth solutions and numerical stability
        - Balanced class weights for potentially imbalanced real-world data
        - Sufficient iterations (1000) to converge on high-dim problems
        
        Suitable for: Raman, FTIR, NIR spectroscopy classification.

        Examples
        --------
        >>> defaults = LogisticRegressionClassifier.default_hyperparams()
        >>> clf = LogisticRegressionClassifier(**defaults)
        """
        return {
            "C": 1.0,
            "penalty": "l2",
            "solver": None,  # Auto-select based on penalty
            "l1_ratio": 0.5,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 0,
            "tol": 1e-4,
        }

    @classmethod
    def sparse_features(cls, **kwargs: Any) -> LogisticRegressionClassifier:
        """Create classifier optimized for sparse feature selection (L1).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LogisticRegressionClassifier
            L1-regularized classifier for sparse solutions.

        Examples
        --------
        >>> clf = LogisticRegressionClassifier.sparse_features(C=0.5)
        >>> # Uses L1 penalty to identify important wavelengths
        """
        params = cls.default_hyperparams()
        params["penalty"] = "l1"
        params["solver"] = "liblinear"  # Only solver supporting L1
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def strong_regularization(cls, **kwargs: Any) -> LogisticRegressionClassifier:
        """Create classifier with strong L2 regularization (low C).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LogisticRegressionClassifier
            Strongly regularized classifier.

        Examples
        --------
        >>> clf = LogisticRegressionClassifier.strong_regularization()
        >>> # Uses C=0.1 for strong regularization on small samples
        """
        params = cls.default_hyperparams()
        params["C"] = 0.1
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def weak_regularization(cls, **kwargs: Any) -> LogisticRegressionClassifier:
        """Create classifier with weak L2 regularization (high C).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LogisticRegressionClassifier
            Weakly regularized classifier.

        Examples
        --------
        >>> clf = LogisticRegressionClassifier.weak_regularization()
        >>> # Uses C=10.0 for weak regularization on large samples
        """
        params = cls.default_hyperparams()
        params["C"] = 10.0
        params.update(kwargs)
        return cls(**params)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to a Joblib file.

        Parameters
        ----------
        path : str or Path
            File path for saving.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self._model, path)

    @classmethod
    def load(cls, path: str | Path) -> LogisticRegressionClassifier:
        """Load a serialized classifier from a Joblib file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        LogisticRegressionClassifier
            Loaded classifier instance marked as fitted.
        """
        model = load(path)
        inst = cls()
        inst._model = model
        inst._fitted = True
        return inst


@dataclass
class LinearSVCClassifier(BaseEstimator):
    """Wrapper over scikit-learn LinearSVC with spectroscopy-optimized defaults.

    Fast linear SVM classifier for high-dimensional data. No probabilistic
    predictions, but exposes decision_function for ranking/ranking tasks.

    Parameters
    ----------
    C : float, default 1.0
        Inverse of regularization strength. Smaller values = stronger regularization.
        Suitable for spectroscopy: values in [0.1, 10.0] work well.

    penalty : {"l1", "l2"}, default "l2"
        Regularization type. L2 is smoother; L1 for sparse solutions.

    dual : bool or "auto", default "auto"
        Choose dual or primal formulation (auto-selects based on n_samples/n_features).

    loss : {"squared_hinge", "hinge"}, default "squared_hinge"
        Loss function. squared_hinge is smoother.

    max_iter : int, default 1000
        Maximum optimization iterations.

    random_state : int, default 0
        Random seed for deterministic training.

    tol : float, default 1e-4
        Convergence tolerance.

    class_weight : {"balanced"} | dict, optional
        Weight adjustment for imbalanced classes.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 500)  # High-dimensional spectroscopy
    >>> y = np.array([0, 1] * 50)
    >>>
    >>> clf = LinearSVCClassifier()
    >>> clf.fit(X, y)
    LinearSVCClassifier(...)
    >>>
    >>> # Get decision values (no probabilities)
    >>> scores = clf.decision_function(X)
    >>> scores.shape
    (100,)
    >>>
    >>> # Use for ranking or threshold-based decisions
    >>> decisions = clf.predict(X)
    >>> decisions.shape
    (100,)

    Notes
    -----
    - **Fast**: Much faster than SVC on large datasets
    - **No probabilities**: Use decision_function for ranking
    - **Spectroscopy**: Good for high-dimensional feature spaces
    - **Sparse features**: Use penalty="l1" for feature selection
    """

    C: float = 1.0
    penalty: Literal["l1", "l2"] = "l2"
    dual: str | bool = "auto"
    loss: Literal["squared_hinge", "hinge"] = "squared_hinge"
    max_iter: int = 1000
    random_state: int = 0
    tol: float = 1e-4
    class_weight: Optional[str | Dict[int, float]] = None

    _model: LinearSVC = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters after dataclass initialization."""
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate parameter combinations."""
        if self.penalty not in ("l1", "l2"):
            raise ValueError(
                f"penalty must be 'l1' or 'l2'; got '{self.penalty}'"
            )

        if self.C <= 0:
            raise ValueError(f"C must be positive; got {self.C}")

        if self.loss not in ("squared_hinge", "hinge"):
            raise ValueError(
                f"loss must be 'squared_hinge' or 'hinge'; got '{self.loss}'"
            )

        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive; got {self.max_iter}")

        if self.tol <= 0:
            raise ValueError(f"tol must be positive; got {self.tol}")

        if self.class_weight is not None:
            if isinstance(self.class_weight, str) and self.class_weight != "balanced":
                raise ValueError(
                    f"class_weight string must be 'balanced' or None; got '{self.class_weight}'"
                )

    def _make_model(self) -> LinearSVC:
        """Create sklearn LinearSVC with validated parameters.

        Returns
        -------
        LinearSVC
            Configured sklearn model.
        """
        kwargs = {
            "C": self.C,
            "penalty": self.penalty,
            "dual": self.dual,
            "loss": self.loss,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "tol": self.tol,
        }

        if self.class_weight is not None:
            kwargs["class_weight"] = self.class_weight

        return LinearSVC(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearSVCClassifier:
        """Fit the linear SVM model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray, shape (n_samples,)
            Training labels.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )

        self._model = self._make_model()
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted class labels.
        """
        self._ensure_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return self._model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function of X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray
            Decision function values. Shape (n_samples,) for binary
            or (n_samples, n_classes) for multiclass.

        Notes
        -----
        LinearSVC does not provide predict_proba. Use decision_function
        for ranking, calibration, or threshold-based decisions.
        """
        self._ensure_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return self._model.decision_function(X)

    def get_coef(self) -> np.ndarray:
        """Get model coefficients.

        Returns
        -------
        ndarray, shape (n_features,) or (n_classes, n_features)
            Model coefficients for each feature.
        """
        self._ensure_fitted()
        coef = self._model.coef_
        # For binary, LinearSVC returns (1, n_features); flatten it
        if coef.shape[0] == 1:
            return coef.ravel()
        return coef

    def get_intercept(self) -> np.ndarray:
        """Get model intercept.

        Returns
        -------
        ndarray or scalar
            Model intercept(s).
        """
        self._ensure_fitted()
        return self._model.intercept_

    @classmethod
    def default_hyperparams(cls) -> Dict[str, Any]:
        """Return spectroscopy-optimized default hyperparameters.

        Returns
        -------
        dict
            Dictionary of default hyperparameters.
        """
        return {
            "C": 1.0,
            "penalty": "l2",
            "dual": "auto",
            "loss": "squared_hinge",
            "max_iter": 1000,
            "random_state": 0,
            "tol": 1e-4,
            "class_weight": "balanced",
        }

    @classmethod
    def sparse_features(cls, **kwargs: Any) -> LinearSVCClassifier:
        """Create classifier with L1 regularization for feature selection.

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LinearSVCClassifier
            L1-regularized classifier.
        """
        params = cls.default_hyperparams()
        params["penalty"] = "l1"
        params["dual"] = False
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def strong_regularization(cls, **kwargs: Any) -> LinearSVCClassifier:
        """Create classifier with strong regularization (small C).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LinearSVCClassifier
            Strongly regularized classifier.
        """
        params = cls.default_hyperparams()
        params["C"] = 0.1
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def weak_regularization(cls, **kwargs: Any) -> LinearSVCClassifier:
        """Create classifier with weak regularization (large C).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        LinearSVCClassifier
            Weakly regularized classifier.
        """
        params = cls.default_hyperparams()
        params["C"] = 10.0
        params.update(kwargs)
        return cls(**params)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to a Joblib file.

        Parameters
        ----------
        path : str or Path
            File path for saving.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self._model, path)

    @classmethod
    def load(cls, path: str | Path) -> LinearSVCClassifier:
        """Load a serialized classifier from a Joblib file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        LinearSVCClassifier
            Loaded classifier instance marked as fitted.
        """
        model = load(path)
        inst = cls()
        inst._model = model
        inst._fitted = True
        return inst


@dataclass
class SVCClassifier(BaseEstimator):
    """Wrapper over scikit-learn SVC with probability support and calibration.

    Non-linear SVM classifier with optional probability estimates and Platt
    calibration. Slower than LinearSVC but offers predict_proba and better
    flexibility for non-linear kernels.

    Parameters
    ----------
    C : float, default 1.0
        Inverse of regularization strength.

    kernel : {"linear", "rbf", "poly", "sigmoid"}, default "rbf"
        Kernel type. RBF is good default for spectroscopy.

    degree : int, default 3
        Degree of polynomial kernel (only used with kernel="poly").

    gamma : {"scale", "auto"} | float, default "scale"
        Kernel coefficient. "scale" = 1/(n_features * variance).

    probability : bool, default True
        If True, enables predict_proba via Platt scaling.

    calibrate : bool, default False
        If True, applies Platt scaling to calibrate probabilities.
        Note: probability=True uses built-in calibration.

    max_iter : int, default 1000
        Maximum optimization iterations.

    random_state : int, default 0
        Random seed for deterministic training.

    tol : float, default 1e-3
        Convergence tolerance.

    class_weight : {"balanced"} | dict, optional
        Weight adjustment for imbalanced classes.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 500)
    >>> y = np.array([0, 1] * 50)
    >>>
    >>> # Default with RBF kernel and probabilities
    >>> clf = SVCClassifier()
    >>> clf.fit(X, y)
    SVCClassifier(...)
    >>>
    >>> # Get probabilistic predictions
    >>> proba = clf.predict_proba(X)
    >>> proba.shape
    (100, 2)
    >>>
    >>> # Linear kernel (faster)
    >>> clf_linear = SVCClassifier(kernel="linear")
    >>> clf_linear.fit(X, y)
    SVCClassifier(kernel='linear', ...)

    Notes
    -----
    - **Slower than LinearSVC**: Due to kernel computation and probability calibration
    - **Flexible kernels**: RBF, poly, sigmoid for non-linear patterns
    - **Probabilities**: Uses Platt scaling for calibration
    - **Spectroscopy**: RBF kernel can capture complex spectral patterns
    """

    C: float = 1.0
    kernel: Literal["linear", "rbf", "poly", "sigmoid"] = "rbf"
    degree: int = 3
    gamma: str | float = "scale"
    probability: bool = True
    calibrate: bool = False
    max_iter: int = 1000
    random_state: int = 0
    tol: float = 1e-3
    class_weight: Optional[str | Dict[int, float]] = None

    _model: SVC | CalibratedClassifierCV = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _has_probability: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters after dataclass initialization."""
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate parameter combinations."""
        if self.kernel not in ("linear", "rbf", "poly", "sigmoid"):
            raise ValueError(
                f"kernel must be 'linear', 'rbf', 'poly', or 'sigmoid'; got '{self.kernel}'"
            )

        if self.C <= 0:
            raise ValueError(f"C must be positive; got {self.C}")

        if self.degree < 1:
            raise ValueError(f"degree must be >= 1; got {self.degree}")

        if isinstance(self.gamma, str) and self.gamma not in ("scale", "auto"):
            raise ValueError(
                f"gamma must be 'scale', 'auto', or float; got '{self.gamma}'"
            )
        elif isinstance(self.gamma, (int, float)) and self.gamma <= 0:
            raise ValueError(f"gamma must be positive; got {self.gamma}")

        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive; got {self.max_iter}")

        if self.tol <= 0:
            raise ValueError(f"tol must be positive; got {self.tol}")

        if self.class_weight is not None:
            if isinstance(self.class_weight, str) and self.class_weight != "balanced":
                raise ValueError(
                    f"class_weight string must be 'balanced' or None; got '{self.class_weight}'"
                )

    def _make_model(self) -> SVC:
        """Create sklearn SVC with validated parameters.

        Returns
        -------
        SVC
            Configured sklearn model.
        """
        kwargs = {
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "probability": self.probability,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "tol": self.tol,
        }

        if self.class_weight is not None:
            kwargs["class_weight"] = self.class_weight

        return SVC(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SVCClassifier:
        """Fit the SVM model with optional Platt calibration.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray, shape (n_samples,)
            Training labels.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )

        base_model = self._make_model()

        # Apply Platt scaling if requested (beyond SVC's built-in calibration)
        if self.calibrate:
            self._model = CalibratedClassifierCV(
                base_model, method="sigmoid", cv=5
            )
        else:
            self._model = base_model

        self._model.fit(X, y)
        self._fitted = True
        self._has_probability = self.probability
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted class labels.
        """
        self._ensure_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples, n_classes)
            Class probabilities.

        Raises
        ------
        RuntimeError
            If probability=False or model not fitted.
        """
        self._ensure_fitted()
        if not self._has_probability:
            raise RuntimeError(
                "predict_proba not available: probability=False. "
                "Create classifier with SVCClassifier(probability=True) to enable probabilities."
            )
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return self._model.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function of X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray
            Decision function values.

        Raises
        ------
        RuntimeError
            If model not fitted.
        """
        self._ensure_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        
        # Handle CalibratedClassifierCV which may not expose decision_function directly
        if isinstance(self._model, CalibratedClassifierCV):
            return self._model.estimator.decision_function(X)
        return self._model.decision_function(X)

    def get_coef(self) -> np.ndarray:
        """Get model coefficients for linear kernel.

        Returns
        -------
        ndarray, shape (n_features,) or (n_classes, n_features)
            Model coefficients.

        Raises
        ------
        AttributeError
            If kernel is not linear.
        """
        self._ensure_fitted()
        if self.kernel != "linear":
            raise ValueError(
                f"Coefficients only available for linear kernel; got {self.kernel}"
            )
        
        model = self._model.estimator if isinstance(self._model, CalibratedClassifierCV) else self._model
        coef = model.coef_
        # For binary, SVC returns (1, n_features); flatten it
        if coef.shape[0] == 1:
            return coef.ravel()
        return coef

    def get_intercept(self) -> np.ndarray:
        """Get model intercept for linear kernel.

        Returns
        -------
        ndarray or scalar
            Model intercept(s).

        Raises
        ------
        AttributeError
            If kernel is not linear.
        """
        self._ensure_fitted()
        if self.kernel != "linear":
            raise ValueError(
                f"Intercept only available for linear kernel; got {self.kernel}"
            )
        
        model = self._model.estimator if isinstance(self._model, CalibratedClassifierCV) else self._model
        return model.intercept_

    @classmethod
    def default_hyperparams(cls) -> Dict[str, Any]:
        """Return spectroscopy-optimized default hyperparameters.

        Returns
        -------
        dict
            Dictionary of default hyperparameters.
        """
        return {
            "C": 1.0,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "probability": True,
            "calibrate": False,
            "max_iter": 1000,
            "random_state": 0,
            "tol": 1e-3,
            "class_weight": "balanced",
        }

    @classmethod
    def linear_kernel(cls, **kwargs: Any) -> SVCClassifier:
        """Create SVM with linear kernel (similar to LinearSVC).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        SVCClassifier
            SVM with linear kernel.
        """
        params = cls.default_hyperparams()
        params["kernel"] = "linear"
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def rbf_kernel(cls, **kwargs: Any) -> SVCClassifier:
        """Create SVM with RBF kernel (default, non-linear).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        SVCClassifier
            SVM with RBF kernel.
        """
        params = cls.default_hyperparams()
        params["kernel"] = "rbf"
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def strong_regularization(cls, **kwargs: Any) -> SVCClassifier:
        """Create classifier with strong regularization (small C).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        SVCClassifier
            Strongly regularized classifier.
        """
        params = cls.default_hyperparams()
        params["C"] = 0.1
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def weak_regularization(cls, **kwargs: Any) -> SVCClassifier:
        """Create classifier with weak regularization (large C).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        SVCClassifier
            Weakly regularized classifier.
        """
        params = cls.default_hyperparams()
        params["C"] = 10.0
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def with_calibration(cls, **kwargs: Any) -> SVCClassifier:
        """Create classifier with Platt calibration for better probability estimates.

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        SVCClassifier
            SVM with calibration enabled.
        """
        params = cls.default_hyperparams()
        params["calibrate"] = True
        params.update(kwargs)
        return cls(**params)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to a Joblib file.

        Parameters
        ----------
        path : str or Path
            File path for saving.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self._model, path)

    @classmethod
    def load(cls, path: str | Path) -> SVCClassifier:
        """Load a serialized classifier from a Joblib file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        SVCClassifier
            Loaded classifier instance marked as fitted.
        """
        model = load(path)
        inst = cls()
        inst._model = model
        inst._fitted = True
        
        # Try to infer probability support from loaded model
        if isinstance(model, CalibratedClassifierCV):
            inst._has_probability = True
        elif hasattr(model, 'probability'):
            inst._has_probability = model.probability
        
        return inst


@dataclass
class RandomForestClassifierWrapper(BaseEstimator):
    """Wrapper over scikit-learn RandomForestClassifier with spectroscopy defaults.

    Fast ensemble classifier for high-dimensional data. Provides feature importance,
    out-of-bag scoring, and probability estimates. Optimized for spectroscopy tasks.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees. More trees = better but slower.
        Suitable for spectroscopy: 100 works well, 200-500 for large datasets.

    max_depth : int, optional
        Maximum tree depth. None = unlimited (can overfit high-dim data).
        Suitable for spectroscopy: 20-30 prevents overfitting on many features.

    min_samples_split : int, default 2
        Minimum samples to split a node. Higher = more regularization.

    min_samples_leaf : int, default 1
        Minimum samples required at leaf. Higher = more regularization.

    max_features : {"sqrt", "log2"} or int, default "sqrt"
        Features considered at each split. "sqrt" reduces correlation for spectroscopy.

    criterion : {"gini", "entropy"}, default "gini"
        Split criterion. Both work similarly; gini is faster.

    class_weight : {"balanced"} | dict, optional
        Weight adjustment for imbalanced classes.

    oob_score : bool, default True
        Use out-of-bag samples to estimate generalization performance.

    n_jobs : int, default -1
        Number of parallel jobs. -1 uses all available cores.

    random_state : int, default 0
        Random seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 500)  # Spectroscopy: many features
    >>> y = np.array([0, 1] * 50)
    >>>
    >>> # Default setup
    >>> clf = RandomForestClassifierWrapper()
    >>> clf.fit(X, y)
    RandomForestClassifierWrapper(...)
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
    >>>
    >>> # Out-of-bag score
    >>> oob_score = clf.oob_score_
    >>> oob_score
    0.92

    Notes
    -----
    - **Fast on large samples**: Ensemble methods scale well
    - **Handles non-linearity**: Captures complex spectral patterns
    - **Feature importance**: Shows which wavelengths matter
    - **Parallelization**: Uses all cores by default (n_jobs=-1)
    - **Out-of-bag scoring**: Free validation via oob_score
    - **Spectroscopy**: Good for 500+ features with many samples
    """

    n_estimators: int = 100
    max_depth: Optional[int] = 20
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int = "sqrt"
    criterion: Literal["gini", "entropy"] = "gini"
    class_weight: Optional[str | Dict[int, float]] = None
    oob_score: bool = True
    n_jobs: int = -1
    random_state: int = 0

    _model: RandomForestClassifier = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters after dataclass initialization."""
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate parameter combinations."""
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive; got {self.n_estimators}")

        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError(f"max_depth must be positive or None; got {self.max_depth}")

        if self.min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2; got {self.min_samples_split}")

        if self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1; got {self.min_samples_leaf}")

        if isinstance(self.max_features, str):
            if self.max_features not in ("sqrt", "log2"):
                raise ValueError(
                    f"max_features string must be 'sqrt' or 'log2'; got '{self.max_features}'"
                )
        elif isinstance(self.max_features, int) and self.max_features <= 0:
            raise ValueError(f"max_features int must be positive; got {self.max_features}")

        if self.criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy'; got '{self.criterion}'"
            )

        if self.class_weight is not None:
            if isinstance(self.class_weight, str) and self.class_weight != "balanced":
                raise ValueError(
                    f"class_weight string must be 'balanced' or None; got '{self.class_weight}'"
                )

    def _make_model(self) -> RandomForestClassifier:
        """Create sklearn RandomForestClassifier with validated parameters.

        Returns
        -------
        RandomForestClassifier
            Configured sklearn model.
        """
        kwargs = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "criterion": self.criterion,
            "oob_score": self.oob_score,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

        if self.class_weight is not None:
            kwargs["class_weight"] = self.class_weight

        return RandomForestClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifierWrapper:
        """Fit the random forest model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray, shape (n_samples,)
            Training labels.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )

        self._model = self._make_model()
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted class labels.
        """
        self._ensure_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples, n_classes)
            Class probabilities.
        """
        self._ensure_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the trained forest.

        Returns
        -------
        ndarray, shape (n_features,)
            Feature importance scores (Gini-based or entropy-based).

        Notes
        -----
        Importances are computed as the decrease in impurity (Gini or entropy)
        caused by each feature. Higher values = more important.
        """
        self._ensure_fitted()
        return self._model.feature_importances_

    @property
    def oob_score_(self) -> float:
        """Out-of-bag score if oob_score=True.

        Returns
        -------
        float
            Out-of-bag accuracy score. Estimates generalization performance.

        Raises
        ------
        RuntimeError
            If oob_score=False or model not fitted.
        """
        self._ensure_fitted()
        if not self.oob_score:
            raise RuntimeError("oob_score not available: oob_score=False")
        return self._model.oob_score_

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
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "criterion": "gini",
            "class_weight": "balanced",
            "oob_score": True,
            "n_jobs": -1,
            "random_state": 0,
        }

    @classmethod
    def fast(cls, **kwargs: Any) -> RandomForestClassifierWrapper:
        """Create classifier for quick iteration (fewer trees).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        RandomForestClassifierWrapper
            Fast classifier with n_estimators=50.
        """
        params = cls.default_hyperparams()
        params["n_estimators"] = 50
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def strong_regularization(cls, **kwargs: Any) -> RandomForestClassifierWrapper:
        """Create classifier with strong regularization (shallow trees).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        RandomForestClassifierWrapper
            Strongly regularized classifier with max_depth=5.
        """
        params = cls.default_hyperparams()
        params["max_depth"] = 5
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def weak_regularization(cls, **kwargs: Any) -> RandomForestClassifierWrapper:
        """Create classifier with weak regularization (deep trees).

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        RandomForestClassifierWrapper
            Weakly regularized classifier with max_depth=None.
        """
        params = cls.default_hyperparams()
        params["max_depth"] = None
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def many_trees(cls, **kwargs: Any) -> RandomForestClassifierWrapper:
        """Create classifier with many trees for better accuracy.

        Parameters
        ----------
        **kwargs
            Override specific hyperparameters.

        Returns
        -------
        RandomForestClassifierWrapper
            High-tree classifier with n_estimators=500.
        """
        params = cls.default_hyperparams()
        params["n_estimators"] = 500
        params.update(kwargs)
        return cls(**params)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to a Joblib file.

        Parameters
        ----------
        path : str or Path
            File path for saving.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self._model, path)

    @classmethod
    def load(cls, path: str | Path) -> RandomForestClassifierWrapper:
        """Load a serialized classifier from a Joblib file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        RandomForestClassifierWrapper
            Loaded classifier instance marked as fitted.
        """
        model = load(path)
        inst = cls()
        inst._model = model
        inst._fitted = True
        return inst


__all__ = [
    "LogisticRegressionClassifier",
    "LinearSVCClassifier",
    "SVCClassifier",
    "RandomForestClassifierWrapper",
]




