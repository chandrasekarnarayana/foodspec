"""Multivariate and dimensionality reduction components."""

from foodspec.multivariate.base import MultivariateComponent, MultivariateResult
from foodspec.multivariate.cca import CCAComponent
from foodspec.multivariate.discriminant import LDAComponent, QDAComponent
from foodspec.multivariate.mds import MDSComponent
from foodspec.multivariate.pca import PCAComponent
from foodspec.multivariate.registry import MULTIVARIATE_REGISTRY, build_component, normalize_method, run_multivariate
from foodspec.multivariate.stats import HotellingT2Component, MANOVAComponent

__all__ = [
    "MultivariateComponent",
    "MultivariateResult",
    "PCAComponent",
    "CCAComponent",
    "LDAComponent",
    "QDAComponent",
    "MDSComponent",
    "HotellingT2Component",
    "MANOVAComponent",
    "MULTIVARIATE_REGISTRY",
    "build_component",
    "run_multivariate",
    "normalize_method",
]
