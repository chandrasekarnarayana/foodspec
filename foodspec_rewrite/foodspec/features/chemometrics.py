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

Chemometrics feature extractors (e.g., PCA).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAFeatureExtractor:
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
    random_state: Optional[int] = 0
    _pca: PCA = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _make_pca(self) -> PCA:
        return PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)

    def fit(self, X: np.ndarray) -> "PCAFeatureExtractor":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if not (1 <= self.n_components <= X.shape[1]):
            raise ValueError("n_components must be in [1, n_features]")
        self._pca = self._make_pca()
        self._pca.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("PCAFeatureExtractor is not fitted; call fit() first on training data.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        Z = self._pca.transform(X)
        cols = [f"pc{i+1}" for i in range(Z.shape[1])]
        return pd.DataFrame(Z, columns=cols)

    def fit_transform(self, X: np.ndarray) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted; call fit() first.")
        return self._pca.explained_variance_ratio_


__all__ = ["PCAFeatureExtractor"]
