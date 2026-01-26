"""Canonical Correlation Analysis component (lightweight)."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA

from foodspec.multivariate.base import MultivariateComponent, MultivariateResult


class CCAComponent(MultivariateComponent):
    method = "cca"
    requires_second_view = True

    def __init__(self, n_components: int = 2, random_state: Optional[int] = None, **kwargs: Any):
        super().__init__(n_components=n_components, random_state=random_state, **kwargs)
        self._random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CCAComponent":
        if y is None:
            raise ValueError("CCA requires a second view array in 'y'.")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.model = CCA(n_components=self.params.get("n_components", 2), random_state=self._random_state)
        self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("CCAComponent not fitted.")
        if y is None:
            raise ValueError("CCA transform requires the paired view 'y'.")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return self.model.transform(X, y)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultivariateResult:
        u, v = self.fit(X, y).transform(X, y)
        scores = np.concatenate([u, v], axis=1)
        return MultivariateResult(
            method=self.method,
            scores=scores,
            loadings=None,
            components=None,
            explained_variance=None,
            params=self.params,
            seed=self._random_state,
            model=self.model,
            metadata={"view1_dim": u.shape[1], "view2_dim": v.shape[1]},
        )


__all__ = ["CCAComponent"]
