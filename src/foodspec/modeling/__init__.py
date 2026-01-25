"""Modeling and validation namespace (mindmap-aligned)."""
from __future__ import annotations

from .models.classical import make_classifier, make_regressor
from .validation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)

__all__ = [
    "make_classifier",
    "make_regressor",
    "compute_classification_metrics",
    "compute_regression_metrics",
]

