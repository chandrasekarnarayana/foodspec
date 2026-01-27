"""Registry for regression / count models (NCSS-inspired)."""
from __future__ import annotations

from typing import Any, Callable, Dict

from sklearn.decomposition import PCA
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    PoissonRegressor,
    RANSACRegressor,
    Ridge,
    TweedieRegressor,
)
from sklearn.pipeline import Pipeline


def _pcr_builder():
    return Pipeline([("pca", PCA()), ("reg", LinearRegression())])


def _neg_binom_builder():
    # Approximate with Tweedie power between Poisson(1) and Gamma(2)
    return TweedieRegressor(power=1.5, link="log")


def _tsls_builder():
    # Simple placeholder: first-stage predicted via LinearRegression inside pipeline hook
    return LinearRegression()


REGRESSION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "linear": {"builder": LinearRegression, "supports_weights": True, "type": "regression"},
    "ridge": {"builder": Ridge, "supports_weights": True, "type": "regression"},
    "huber": {"builder": HuberRegressor, "supports_weights": False, "type": "regression"},
    "ransac": {"builder": lambda: RANSACRegressor(base_estimator=LinearRegression()), "supports_weights": False, "type": "regression"},
    "pcr": {"builder": _pcr_builder, "supports_weights": False, "type": "regression"},
    "poisson": {"builder": lambda: PoissonRegressor(alpha=0.0), "supports_weights": True, "type": "count"},
    "neg_binom": {"builder": _neg_binom_builder, "supports_weights": True, "type": "count"},
    "tsls": {"builder": _tsls_builder, "supports_weights": False, "type": "regression", "note": "simple 2SLS placeholder"},
}


def build_regression_model(name: str):
    key = name.strip().lower().replace("-", "_")
    if key not in REGRESSION_REGISTRY:
        raise ValueError(f"Unknown regression model '{name}'.")
    builder: Callable[[], Any] = REGRESSION_REGISTRY[key]["builder"]
    return builder()


__all__ = ["REGRESSION_REGISTRY", "build_regression_model"]
