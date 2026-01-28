"""Linear / Quadratic Discriminant Analysis as projections."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from foodspec.multivariate.base import MultivariateComponent, MultivariateResult


class LDAComponent(MultivariateComponent):
    method = "lda"
    requires_y = True

    def __init__(self, solver: str = "svd", **kwargs: Any):
        super().__init__(solver=solver, **kwargs)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LDAComponent":
        if y is None:
            raise ValueError("LDA requires class labels y.")
        self.model = LinearDiscriminantAnalysis(solver=self.params.get("solver", "svd"))
        self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LDAComponent not fitted.")
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultivariateResult:
        scores = self.fit(X, y).transform(X)
        return MultivariateResult(
            method=self.method,
            scores=scores,
            params=self.params,
            model=self.model,
        )


class QDAComponent(MultivariateComponent):
    method = "qda"
    requires_y = True

    def __init__(self, reg_param: float = 0.0, **kwargs: Any):
        super().__init__(reg_param=reg_param, **kwargs)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "QDAComponent":
        if y is None:
            raise ValueError("QDA requires class labels y.")
        self.model = QuadraticDiscriminantAnalysis(reg_param=self.params.get("reg_param", 0.0))
        self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("QDAComponent not fitted.")
        return self.model.predict_proba(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultivariateResult:
        scores = self.fit(X, y).transform(X)
        return MultivariateResult(
            method=self.method,
            scores=scores,
            params=self.params,
            model=self.model,
        )


__all__ = ["LDAComponent", "QDAComponent"]
