"""Modeling and validation namespace (mindmap-aligned)."""
from __future__ import annotations

from .api import FitPredictResult, fit_predict, metrics_by_group
from .models.classical import make_classifier, make_regressor
from .models_regression import REGRESSION_REGISTRY, build_regression_model
from .outcome import OutcomeType
from .validation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)

__all__ = [
    "make_classifier",
    "make_regressor",
    "FitPredictResult",
    "fit_predict",
    "metrics_by_group",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "OutcomeType",
    "REGRESSION_REGISTRY",
    "build_regression_model",
]
