"""Regression and count metrics plus diagnostics."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn import metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    mae = float(metrics.mean_absolute_error(y_true, y_pred))
    r2 = float(metrics.r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))))
    return {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias, "mape": mape}


def count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    mae = float(metrics.mean_absolute_error(y_true, y_pred))
    # Poisson deviance proxy (clip to avoid log(0))
    eps = 1e-8
    y_pred_clip = np.clip(y_pred, eps, None)
    dev = 2.0 * np.sum(y_true * np.log((y_true + eps) / y_pred_clip) - (y_true - y_pred_clip))
    overdisp = float(np.var(y_true) / (np.mean(y_true) + eps)) if np.mean(y_true) > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "poisson_deviance": float(dev / len(y_true)), "overdispersion": overdisp}


def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    abs_resid = np.abs(resid)
    hetero = float(np.corrcoef(abs_resid, np.abs(y_pred))[0, 1]) if len(resid) > 1 else 0.0
    return {
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid)),
        "heteroscedasticity_corr": hetero,
    }


def overdispersion_summary(y_true: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    mean = float(np.mean(y_true)) if len(y_true) else 0.0
    var = float(np.var(y_true)) if len(y_true) else 0.0
    ratio = var / (mean + 1e-8) if mean > 0 else 0.0
    return ratio, mean


__all__ = [
    "regression_metrics",
    "count_metrics",
    "residual_diagnostics",
    "overdispersion_summary",
]
