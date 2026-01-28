"""Principal Component Analysis component."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA

from foodspec.multivariate.base import MultivariateComponent, MultivariateResult


class PCAComponent(MultivariateComponent):
    method = "pca"

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(n_components=n_components, whiten=whiten, random_state=random_state, **kwargs)
        self._random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCAComponent":
        self.model = PCA(
            n_components=self.params.get("n_components"),
            whiten=bool(self.params.get("whiten", False)),
            random_state=self._random_state,
        )
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCAComponent not fitted.")
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultivariateResult:
        self.fit(X, y)
        scores = self.model.transform(X)
        loadings = getattr(self.model, "components_", None)
        explained = getattr(self.model, "explained_variance_ratio_", None)
        return MultivariateResult(
            method=self.method,
            scores=scores,
            loadings=loadings,
            components=loadings,
            explained_variance=explained,
            params=self.params,
            seed=self._random_state,
            model=self.model,
        )


__all__ = ["PCAComponent"]
