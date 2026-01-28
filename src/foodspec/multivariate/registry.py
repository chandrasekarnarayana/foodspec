"""Registry and helpers for multivariate components."""

from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np

from foodspec.multivariate.base import MultivariateComponent, MultivariateResult
from foodspec.multivariate.cca import CCAComponent
from foodspec.multivariate.discriminant import LDAComponent, QDAComponent
from foodspec.multivariate.mds import MDSComponent
from foodspec.multivariate.pca import PCAComponent
from foodspec.multivariate.stats import HotellingT2Component, MANOVAComponent

MULTIVARIATE_REGISTRY: Dict[str, Type[MultivariateComponent]] = {
    "pca": PCAComponent,
    "cca": CCAComponent,
    "lda": LDAComponent,
    "qda": QDAComponent,
    "mds": MDSComponent,
    "hotelling_t2": HotellingT2Component,
    "manova": MANOVAComponent,
}


def normalize_method(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def build_component(method: str, **params: Any) -> MultivariateComponent:
    key = normalize_method(method)
    if key not in MULTIVARIATE_REGISTRY:
        raise ValueError(f"Unknown multivariate method '{method}'. Available: {sorted(MULTIVARIATE_REGISTRY)}")
    cls = MULTIVARIATE_REGISTRY[key]
    return cls(**params)


def run_multivariate(method: str, X: np.ndarray, y: Any = None, **params: Any) -> MultivariateResult:
    component = build_component(method, **params)
    return component.fit_transform(X, y)


__all__ = ["MULTIVARIATE_REGISTRY", "build_component", "run_multivariate", "normalize_method"]
