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

Base classes and interfaces for FoodSpec models.
Provides standard sklearn-like API with save/load, parameter management, and label encoding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder


class LabelEncoderWrapper:
    """Wrapper for consistent and reproducible label encoding.

    Ensures that class labels are encoded consistently across folds
    and that class order is deterministic. Handles both binary and
    multi-class classification.

    Examples
    --------
    >>> import numpy as np
    >>> encoder = LabelEncoderWrapper()
    >>> y = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])
    >>> y_encoded = encoder.fit_transform(y)
    >>> y_encoded
    array([0, 1, 0, 2, 1])
    >>> encoder.classes_
    array(['bird', 'cat', 'dog'], dtype=object)
    >>> y_new = np.array(['dog', 'cat'])
    >>> encoder.transform(y_new)
    array([1, 0])
    """

    def __init__(self) -> None:
        self._encoder: Optional[LabelEncoder] = None
        self._fitted: bool = False

    @property
    def classes_(self) -> np.ndarray:
        """Get the learned class labels."""
        if not self._fitted:
            raise RuntimeError("LabelEncoder not fitted; call fit_transform() first")
        assert self._encoder is not None
        return self._encoder.classes_

    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(self.classes_)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit encoder and transform labels to integers [0, n_classes)."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        
        # Sort classes for deterministic ordering
        self._encoder = LabelEncoder()
        self._encoder.fit(y)
        self._fitted = True
        return self._encoder.transform(y)

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels using the fitted encoder."""
        if not self._fitted:
            raise RuntimeError("LabelEncoder not fitted; call fit_transform() first")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        assert self._encoder is not None
        return self._encoder.transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Transform encoded labels back to original class labels."""
        if not self._fitted:
            raise RuntimeError("LabelEncoder not fitted; call fit_transform() first")
        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        assert self._encoder is not None
        return self._encoder.inverse_transform(y)


class BaseEstimator(ABC):
    """Base interface for all FoodSpec models.

    Provides sklearn-like API with fit/predict/predict_proba,
    parameter management, and serialization support.
    
    All subclasses must:
    1. Implement fit(X, y) and predict(X)
    2. Optionally implement predict_proba(X)
    3. Support get_params() and set_params() automatically
    4. Support clone_with_params() for parameter updates
    5. Support save(path) and load(path) for serialization

    Examples
    --------
    >>> from foodspec.models.base import BaseEstimator
    >>> import numpy as np
    >>>
    >>> # Create a custom model
    >>> class CustomModel(BaseEstimator):
    ...     def __init__(self, alpha=1.0):
    ...         self.alpha = alpha
    ...     def fit(self, X, y):
    ...         self._fitted = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.zeros(X.shape[0])
    ...     def predict_proba(self, X):
    ...         return np.ones((X.shape[0], 2)) * 0.5
    >>>
    >>> model = CustomModel(alpha=2.0)
    >>> X = np.random.randn(10, 5)
    >>> y = np.array([0, 1] * 5)
    >>> model.fit(X, y)
    CustomModel(alpha=2.0)
    >>> model.predict(X).shape
    (10,)
    >>> model.predict_proba(X).shape
    (10, 2)
    """

    def __init__(self) -> None:
        self._fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Fit the model to training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray, shape (n_samples,)
            Training labels.

        Returns
        -------
        self
            Returns the fitted estimator for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted class labels.
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

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
        NotImplementedError
            If the model does not support probabilistic predictions.
        """
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not support predict_proba(). "
            "Use a model with probabilistic predictions or implement this method."
        )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters.

        Parameters
        ----------
        deep : bool, default True
            If True, recursively get nested object parameters.

        Returns
        -------
        dict
            Dictionary mapping parameter names to values.

        Examples
        --------
        >>> from foodspec.models.classical import LogisticRegressionClassifier
        >>> model = LogisticRegressionClassifier(C=2.0, max_iter=500)
        >>> params = model.get_params()
        >>> params['C']
        2.0
        >>> params['max_iter']
        500
        """
        params: Dict[str, Any] = {}
        
        # Get all non-private attributes from the instance
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if deep and isinstance(value, BaseEstimator):
                    # Recursively get nested estimator params
                    nested_params = value.get_params(deep=True)
                    for nested_key, nested_val in nested_params.items():
                        params[f"{key}__{nested_key}"] = nested_val
                else:
                    params[key] = value
        
        return params

    def set_params(self, **params: Any) -> BaseEstimator:
        """Set model parameters.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If a parameter name is invalid.

        Examples
        --------
        >>> from foodspec.models.classical import LogisticRegressionClassifier
        >>> model = LogisticRegressionClassifier()
        >>> model.set_params(C=5.0, max_iter=2000)  # doctest: +SKIP
        LogisticRegressionClassifier(...)
        >>> model.C  # doctest: +SKIP
        5.0
        """
        if not params:
            return self
        
        valid_params = self.get_params(deep=True)
        
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for model {self.__class__.__name__}. "
                    f"Valid parameters: {list(valid_params.keys())}"
                )
            
            # Handle nested parameters (e.g., "estimator__param_name")
            if "__" in key:
                obj_name, param_name = key.split("__", 1)
                if hasattr(self, obj_name):
                    obj = getattr(self, obj_name)
                    if isinstance(obj, BaseEstimator):
                        obj.set_params(**{param_name: value})
                    else:
                        setattr(self, obj_name, value)
            else:
                setattr(self, key, value)
        
        return self

    def clone_with_params(self, **params: Any) -> BaseEstimator:
        """Create a copy of this estimator with updated parameters.

        Does not modify the original estimator.

        Parameters
        ----------
        **params : dict
            New parameter values.

        Returns
        -------
        BaseEstimator
            A new estimator instance with updated parameters.

        Examples
        --------
        >>> from foodspec.models.classical import LogisticRegressionClassifier
        >>> model = LogisticRegressionClassifier(C=1.0)
        >>> model2 = model.clone_with_params(C=10.0)
        >>> model.C
        1.0
        >>> model2.C
        10.0
        """
        cloned = deepcopy(self)
        if params:
            cloned.set_params(**params)
        return cloned

    def save(self, path: str | Path) -> None:
        """Save the fitted estimator to a file using joblib.

        Parameters
        ----------
        path : str or Path
            File path for saving.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If path is invalid.

        Examples
        --------
        >>> from foodspec.models.classical import LogisticRegressionClassifier
        >>> import numpy as np
        >>> X = np.random.randn(10, 5)
        >>> y = np.array([0, 1] * 5)
        >>> model = LogisticRegressionClassifier()
        >>> model.fit(X, y)
        LogisticRegressionClassifier(...)
        >>> model.save("/tmp/my_model.joblib")  # doctest: +SKIP
        """
        if not self._fitted:
            raise RuntimeError(
                f"Cannot save unfitted model {self.__class__.__name__}. "
                "Call fit() first."
            )
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseEstimator:
        """Load a fitted estimator from a file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        BaseEstimator
            The loaded estimator.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a valid joblib model.

        Examples
        --------
        >>> from foodspec.models.classical import LogisticRegressionClassifier
        >>> model = LogisticRegressionClassifier.load("/tmp/my_model.joblib")  # doctest: +SKIP
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            model = load(path)
            if not isinstance(model, BaseEstimator):
                raise ValueError(
                    f"Loaded object is not a BaseEstimator; got {type(model).__name__}"
                )
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {path}: {e}") from e

    def _ensure_fitted(self) -> None:
        """Check that the model is fitted.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                f"Model {self.__class__.__name__} is not fitted. Call fit() first."
            )


__all__ = ["BaseEstimator", "LabelEncoderWrapper"]
