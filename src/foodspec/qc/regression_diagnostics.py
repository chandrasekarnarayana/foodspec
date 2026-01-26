"""Lightweight regression/count diagnostics for QC artifacts."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from foodspec.modeling.metrics_regression import residual_diagnostics, overdispersion_summary
from foodspec.modeling.outcome import OutcomeType


def summarize_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    outcome_type: OutcomeType | str = OutcomeType.REGRESSION,
    max_rows: int = 20,
) -> Dict[str, Any]:
    """Summarize residuals and simple QC flags.

    Returns summary stats, flags, and a small residuals table for reporting.
    """

    outcome_enum = OutcomeType(outcome_type) if isinstance(outcome_type, str) else outcome_type
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    resid = yt - yp

    summary = residual_diagnostics(yt, yp)
    flags: List[str] = []

    if abs(summary.get("residual_mean", 0.0)) > max(1e-6, 0.1 * (np.std(yt) + 1e-8)):
        flags.append("residual_bias_large")
    if abs(summary.get("heteroscedasticity_corr", 0.0)) > 0.6:
        flags.append("heteroscedasticity_detected")

    if outcome_enum == OutcomeType.COUNT:
        ratio, mean_count = overdispersion_summary(yt)
        summary["overdispersion_ratio"] = ratio
        summary["mean_count"] = mean_count
        if ratio > 1.5:
            flags.append("overdispersion_high")

    residuals_df = pd.DataFrame({
        "y_true": yt,
        "y_pred": yp,
        "residual": resid,
    })
    residuals_df["abs_residual"] = residuals_df["residual"].abs()
    residuals_df.sort_values("abs_residual", ascending=False, inplace=True)
    residuals_top = residuals_df.head(max_rows).reset_index(drop=True)

    return {
        "summary": summary,
        "flags": flags,
        "residuals": residuals_top,
    }


__all__ = ["summarize_regression_diagnostics"]