"""Modeling and validation namespace (mindmap-aligned)."""
from __future__ import annotations

from .api import FitPredictResult, compute_roc_for_result, fit_predict, metrics_by_group
from .diagnostics import (
    RocDiagnosticsResult,
    compute_roc_diagnostics,
)
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
    "compute_roc_for_result",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "OutcomeType",
    "REGRESSION_REGISTRY",
    "build_regression_model",
    "compute_roc_diagnostics",
    "RocDiagnosticsResult",
]
