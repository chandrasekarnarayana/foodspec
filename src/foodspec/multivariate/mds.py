"""Multidimensional scaling component."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.manifold import MDS

from foodspec.multivariate.base import MultivariateComponent, MultivariateResult


class MDSComponent(MultivariateComponent):
    method = "mds"

    def __init__(self, n_components: int = 2, metric: bool = True, random_state: Optional[int] = None, **kwargs: Any):
        super().__init__(n_components=n_components, metric=metric, random_state=random_state, **kwargs)
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MDSComponent":
        self.model = MDS(
            n_components=self.params.get("n_components", 2),
            metric=self.params.get("metric", True),
            random_state=self.params.get("random_state", None),
            normalized_stress="auto",
        )
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MDSComponent not fitted.")
        return self.model.embedding_

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultivariateResult:
        model = MDS(
            n_components=self.params.get("n_components", 2),
            metric=self.params.get("metric", True),
            random_state=self.params.get("random_state", None),
            normalized_stress="auto",
        )
        scores = model.fit_transform(X)
        self.model = model
        return MultivariateResult(
            method=self.method,
            scores=scores,
            params=self.params,
            model=self.model,
        )


__all__ = ["MDSComponent"]
