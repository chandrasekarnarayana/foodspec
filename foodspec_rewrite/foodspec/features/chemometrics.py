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

Chemometrics feature extractors (PCA, PLS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from .base import FeatureExtractor


@dataclass
class PCAFeatureExtractor(FeatureExtractor):
    """Principal Components feature extractor using scikit-learn PCA.

    Parameters
    ----------
    n_components : int, default 2
        Number of principal components to retain.
    whiten : bool, default False
        If True, components are scaled to unit variance.
    random_state : int | None, default 0
        Seed for deterministic behavior of PCA when applicable.

    Notes
    -----
    - Call `fit(X)` on training data only; then `transform(X)` on any split.
    - `fit_transform(X)` is provided for convenience on the same split.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.vstack([np.linspace(0, 1, 10), np.linspace(1, 0, 10)]).astype(float)
    >>> pca = PCAFeatureExtractor(n_components=2, random_state=0)
    >>> pca.fit(X)
    PCAFeatureExtractor(n_components=2, whiten=False, random_state=0)
    >>> Z = pca.transform(X)
    >>> Z.shape
    (2, 2)
    """

    n_components: int = 2
    whiten: bool = False
    seed: Optional[int] = None
    _pca: PCA = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _make_pca(self) -> PCA:
        # sklearn PCA uses `random_state` for some solver paths; map from `seed`
        return PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.seed)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "PCAFeatureExtractor":
        """Fit PCA on data (y and kwargs ignored for unsupervised PCA)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if not (1 <= self.n_components <= X.shape[1]):
            raise ValueError("n_components must be in [1, n_features]")
        self._pca = self._make_pca()
        self._pca.fit(X)
        self._fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if not self._fitted:
            raise RuntimeError("PCAFeatureExtractor is not fitted; call fit() first on training data.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        Z = self._pca.transform(X)
        cols = [f"pca_{i+1}" for i in range(Z.shape[1])]
        return Z, cols

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform in one call."""
        self.fit(X, y, x_grid, meta)
        return self.transform(X, x_grid, meta)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted; call fit() first.")
        return self._pca.explained_variance_ratio_


@dataclass
class PLSFeatureExtractor(FeatureExtractor):
    """Partial Least Squares feature extractor for supervised dimensionality reduction.

    PLS finds latent variables that maximize covariance between X and y, making it
    suitable for feature extraction when labels are available during training.

    Parameters
    ----------
    n_components : int, default 2
        Number of PLS components to retain.
    scale : bool, default True
        If True, scale X to unit variance before fitting.
    random_state : int | None, default 0
        Seed for deterministic behavior (used if sklearn PLS supports it).

    Notes
    -----
    - MUST call fit(X, y) on training data with labels
    - Then transform(X) on any split (test/validation)
    - Leakage-safe: fitting is done ONLY on training fold

    Examples
    --------
    >>> import numpy as np
    >>> X_train = np.random.randn(50, 100)
    >>> y_train = np.random.randint(0, 2, 50)
    >>> X_test = np.random.randn(20, 100)
    >>> pls = PLSFeatureExtractor(n_components=3, scale=True)
    >>> pls.fit(X_train, y_train)
    PLSFeatureExtractor(n_components=3, scale=True, random_state=0)
    >>> Z_train = pls.transform(X_train)
    >>> Z_test = pls.transform(X_test)
    >>> Z_train.shape
    (50, 3)
    >>> Z_test.shape
    (20, 3)
    """

    n_components: int = 2
    mode: str = "regression"  # or "classification"
    scale: bool = True
    _pls: PLSRegression = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _make_pls(self) -> PLSRegression:
        return PLSRegression(n_components=self.n_components, scale=self.scale)

    @staticmethod
    def _encode_labels(y: np.ndarray) -> np.ndarray:
        """Encode arbitrary labels to numeric codes deterministically.

        Uses sorted unique labels to assign codes 0..K-1.
        """
        if y.dtype.kind in {"i", "u", "f"}:
            return y.astype(float)
        uniques, inverse = np.unique(y, return_inverse=True)
        return inverse.astype(float)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "PLSFeatureExtractor":
        """Fit PLS on training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training spectral data.
        y : ndarray, shape (n_samples,)
            Training labels (required for PLS).

        Returns
        -------
        self : PLSFeatureExtractor
            Fitted extractor.

        Raises
        ------
        ValueError
            If y is None or shapes mismatch.
        """
        if y is None:
            raise ValueError("PLSFeatureExtractor requires y (labels) for fitting")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"y length ({y.shape[0]}) must match X samples ({X.shape[0]})")
        if not (1 <= self.n_components <= min(X.shape)):
            raise ValueError(f"n_components must be in [1, min(n_samples, n_features)], got {self.n_components}")

        # Handle mode
        if self.mode not in ("regression", "classification"):
            raise ValueError("mode must be 'regression' or 'classification'")

        self._pls = self._make_pls()
        # Prepare y
        if self.mode == "classification":
            y_num = self._encode_labels(y)
        else:
            # regression: must be numeric
            if y.dtype.kind not in {"i", "u", "f"}:
                raise ValueError("For regression mode, y must be numeric")
            y_num = y.astype(float)

        y_2d = y_num.reshape(-1, 1) if y_num.ndim == 1 else y_num
        self._pls.fit(X, y_2d)
        self._fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Transform data to PLS component space.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to transform (can be train or test).

        Returns
        -------
        features : DataFrame
            PLS components with columns [pls1, pls2, ...].

        Raises
        ------
        RuntimeError
            If transform called before fit.
        """
        if not self._fitted:
            raise RuntimeError("PLSFeatureExtractor not fitted; call fit(X, y) first on training data")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        Z = self._pls.transform(X)
        cols = [f"pls_{i+1}" for i in range(Z.shape[1])]
        return Z, cols

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        x_grid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform in one call.

        Parameters
        ----------
        X : ndarray
            Training data.
        y : ndarray
            Training labels (required).

        Returns
        -------
        features : DataFrame
            PLS components for training data.
        """
        self.fit(X, y, x_grid, meta)
        return self.transform(X, x_grid, meta)


__all__ = ["PCAFeatureExtractor", "PLSFeatureExtractor"]
