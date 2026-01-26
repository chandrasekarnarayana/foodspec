"""Base interfaces and result dataclasses for multivariate analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class MultivariateResult:
    """Container for multivariate analysis outputs."""

    method: str
    scores: np.ndarray
    loadings: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None
    distances: Optional[np.ndarray] = None
    params: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    model: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, Any] = field(default_factory=dict)


class MultivariateComponent:
    """Interface for multivariate components."""

    method: str = "base"
    requires_y: bool = False
    requires_second_view: bool = False

    def __init__(self, **params: Any):
        self.params = params
        self.model = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MultivariateComponent":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultivariateResult:
        self.fit(X, y)
        scores = self.transform(X)
        return MultivariateResult(method=self.method, scores=scores, params=self.params, model=self.model)


__all__ = ["MultivariateResult", "MultivariateComponent"]
