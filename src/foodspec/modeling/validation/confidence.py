"""Confidence interval helpers (shim to foodspec.chemometrics.validation)."""

from __future__ import annotations

from foodspec.modeling.validation.metrics import bootstrap_prediction_intervals

__all__ = ["bootstrap_prediction_intervals"]
