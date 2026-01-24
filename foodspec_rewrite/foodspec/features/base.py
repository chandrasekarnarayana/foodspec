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

Feature engineering base classes and containers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FeatureExtractor(ABC):
    """Base class for feature extractors with sklearn-compatible API.

    All feature extractors must inherit from this class to ensure:
    - Leakage safety: fit() called ONLY on training data
    - Consistent interface: fit/transform/fit_transform pattern
    - Parameter management: get_params/set_params for sklearn compatibility

    Parameters
    ----------
    Any subclass-specific parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from foodspec.features.chemometrics import PCAFeatureExtractor
    >>> X_train = np.random.randn(50, 100)
    >>> X_test = np.random.randn(20, 100)
    >>> extractor = PCAFeatureExtractor(n_components=5)
    >>> extractor.fit(X_train)
    PCAFeatureExtractor(n_components=5, whiten=False, random_state=0)
    >>> Xf, names = extractor.transform(X_test)
    >>> Xf.shape
    (20, 5)
    >>> len(names)
    5
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "FeatureExtractor":
        """Fit the extractor on training data ONLY.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data (spectral matrix or feature matrix).
        y : ndarray, shape (n_samples,), optional
            Training labels for supervised methods (e.g., PLS).
        x_grid : ndarray, shape (n_features,), optional
            Wavenumber/wavelength grid for peak/band extractors.
        meta : dict, optional
            Additional metadata (e.g., sample IDs, experimental conditions).

        Returns
        -------
        self : FeatureExtractor
            Fitted extractor instance for method chaining.
        """
        ...

    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Transform data to features after fitting.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to transform (can be train, validation, or test).
        x_grid : ndarray, shape (n_features,), optional
            Wavenumber grid (must match fit if provided).
        meta : dict, optional
            Additional metadata.

        Returns
        -------
        Xf : ndarray, shape (n_samples, n_features_out)
            Extracted features as numpy array.
        feature_names : list of str
            Names of extracted features (length must equal n_features_out).
        """
        ...

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform in one call (convenience method).

        Parameters
        ----------
        X : ndarray
            Training data.
        y : ndarray, optional
            Training labels for supervised methods.
        x_grid : ndarray, optional
            Wavenumber grid.
        meta : dict, optional
            Additional metadata.

        Returns
        -------
        Xf : ndarray
            Extracted features.
        feature_names : list of str
            Feature names.
        """
        self.fit(X, y, x_grid, meta)
        return self.transform(X, x_grid, meta)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (sklearn compatibility).

        Parameters
        ----------
        deep : bool, default True
            If True, return parameters for nested objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        # Default implementation: return all public attributes set in __init__
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def set_params(self, **params: Any) -> "FeatureExtractor":
        """Set parameters for this estimator (sklearn compatibility).

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : FeatureExtractor
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


@dataclass
class FeatureSet:
    """Container for extracted features with metadata.

    Stores features as numpy array with associated names and optional metadata
    about peaks, bands, or other extraction parameters.

    Attributes
    ----------
    Xf : ndarray, shape (n_samples, n_features)
        Extracted feature matrix.
    feature_names : list of str
        Ordered list of feature names (length must equal n_features).
    feature_meta : dict, optional
        Additional metadata about features (e.g., peak locations, band ranges,
        explained variance ratios, extractor names).

    Examples
    --------
    >>> import numpy as np
    >>> Xf = np.random.randn(10, 3)
    >>> names = ["pc1", "pc2", "pc3"]
    >>> meta = {"explained_variance_ratio": [0.45, 0.23, 0.12]}
    >>> fs = FeatureSet(Xf=Xf, feature_names=names, feature_meta=meta)
    >>> fs.n_samples
    10
    >>> fs.n_features
    3
    >>> fs.feature_names
    ['pc1', 'pc2', 'pc3']
    """

    Xf: np.ndarray
    feature_names: List[str]
    feature_meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate shapes and consistency."""
        if not isinstance(self.Xf, np.ndarray):
            raise TypeError(f"Xf must be numpy array, got {type(self.Xf)}")
        
        if self.Xf.ndim != 2:
            raise ValueError(f"Xf must be 2D (n_samples, n_features), got shape {self.Xf.shape}")
        
        if not isinstance(self.feature_names, list):
            raise TypeError(f"feature_names must be list, got {type(self.feature_names)}")
        
        if len(self.feature_names) != self.Xf.shape[1]:
            raise ValueError(
                f"Length of feature_names ({len(self.feature_names)}) must match "
                f"n_features ({self.Xf.shape[1]})"
            )

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.Xf.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.Xf.shape[1]

    def select_features(self, indices: List[int]) -> "FeatureSet":
        """Select subset of features by index.

        Parameters
        ----------
        indices : list of int
            Feature indices to keep.

        Returns
        -------
        subset : FeatureSet
            New FeatureSet with selected features only.

        Examples
        --------
        >>> import numpy as np
        >>> Xf = np.random.randn(10, 5)
        >>> names = ["f1", "f2", "f3", "f4", "f5"]
        >>> fs = FeatureSet(Xf=Xf, feature_names=names)
        >>> subset = fs.select_features([0, 2, 4])
        >>> subset.n_features
        3
        >>> subset.feature_names
        ['f1', 'f3', 'f5']
        """
        selected_Xf = self.Xf[:, indices]
        selected_names = [self.feature_names[i] for i in indices]
        return FeatureSet(
            Xf=selected_Xf,
            feature_names=selected_names,
            feature_meta=self.feature_meta.copy(),
        )

    def concatenate(self, other: "FeatureSet") -> "FeatureSet":
        """Concatenate features horizontally with another FeatureSet.

        Parameters
        ----------
        other : FeatureSet
            Another FeatureSet with same n_samples.

        Returns
        -------
        combined : FeatureSet
            New FeatureSet with features concatenated horizontally.

        Raises
        ------
        ValueError
            If n_samples don't match.

        Examples
        --------
        >>> import numpy as np
        >>> fs1 = FeatureSet(np.random.randn(10, 2), ["a", "b"])
        >>> fs2 = FeatureSet(np.random.randn(10, 3), ["c", "d", "e"])
        >>> combined = fs1.concatenate(fs2)
        >>> combined.n_features
        5
        >>> combined.feature_names
        ['a', 'b', 'c', 'd', 'e']
        """
        if self.n_samples != other.n_samples:
            raise ValueError(
                f"Cannot concatenate: n_samples mismatch ({self.n_samples} vs {other.n_samples})"
            )

        combined_Xf = np.hstack([self.Xf, other.Xf])
        combined_names = self.feature_names + other.feature_names
        combined_meta = {**self.feature_meta, **other.feature_meta}

        return FeatureSet(
            Xf=combined_Xf,
            feature_names=combined_names,
            feature_meta=combined_meta,
        )


__all__ = ["FeatureExtractor", "FeatureSet"]
