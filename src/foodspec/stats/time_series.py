from __future__ import annotations
"""
Time series forecasting utilities for drift and stability monitoring.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class ForecastResult:
    forecast: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    aic: Optional[float] = None
    bic: Optional[float] = None


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation up to max_lag (inclusive)."""
    y = np.asarray(series, dtype=float)
    y = y - np.mean(y)
    result = np.correlate(y, y, mode="full")
    mid = result.size // 2
    acf = result[mid : mid + max_lag + 1] / result[mid]
    return acf


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute cross-correlation between two series."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    corr = np.correlate(x, y, mode="full")
    mid = corr.size // 2
    return corr[mid - max_lag : mid + max_lag + 1]


def spectral_analysis(series: np.ndarray, *, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density via periodogram."""
    y = np.asarray(series, dtype=float)
    freqs, psd = signal.periodogram(y, fs=fs)
    return freqs, psd


def fit_arima(
    series: np.ndarray,
    order: tuple[int, int, int],
    *,
    forecast_steps: int = 5,
) -> ForecastResult:
    """Fit ARIMA using statsmodels and return forecast."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # pragma: no cover
        raise ImportError("statsmodels is required for ARIMA.") from exc

    y = np.asarray(series, dtype=float)
    model = ARIMA(y, order=order).fit()
    forecast = model.forecast(steps=forecast_steps)
    fitted = model.fittedvalues
    residuals = y - fitted
    return ForecastResult(
        forecast=np.asarray(forecast, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        aic=float(model.aic),
        bic=float(model.bic),
    )


def fit_exponential_smoothing(
    series: np.ndarray,
    *,
    trend: Optional[str] = None,
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None,
    forecast_steps: int = 5,
) -> ForecastResult:
    """Fit exponential smoothing (Holt-Winters) model."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception as exc:  # pragma: no cover
        raise ImportError("statsmodels is required for exponential smoothing.") from exc

    y = np.asarray(series, dtype=float)
    model = ExponentialSmoothing(
        y,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    ).fit(optimized=True)
    forecast = model.forecast(forecast_steps)
    fitted = model.fittedvalues
    residuals = y - fitted
    return ForecastResult(
        forecast=np.asarray(forecast, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        aic=float(model.aic) if hasattr(model, "aic") else None,
        bic=float(model.bic) if hasattr(model, "bic") else None,
    )


__all__ = [
    "ForecastResult",
    "autocorrelation",
    "cross_correlation",
    "spectral_analysis",
    "fit_arima",
    "fit_exponential_smoothing",
]
