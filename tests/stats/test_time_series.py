from __future__ import annotations

import numpy as np

from foodspec.stats.time_series import (
    autocorrelation,
    cross_correlation,
    fit_arima,
    fit_exponential_smoothing,
    spectral_analysis,
)


def test_autocorrelation_shapes() -> None:
    series = np.arange(10, dtype=float)
    acf = autocorrelation(series, max_lag=4)
    assert acf.shape[0] == 5


def test_cross_correlation_shapes() -> None:
    x = np.arange(10, dtype=float)
    y = x + 0.1
    ccf = cross_correlation(x, y, max_lag=3)
    assert ccf.shape[0] == 7


def test_spectral_analysis() -> None:
    series = np.sin(np.linspace(0, 2 * np.pi, 50))
    freqs, psd = spectral_analysis(series)
    assert freqs.shape == psd.shape


def test_fit_arima() -> None:
    series = np.sin(np.linspace(0, 2 * np.pi, 30))
    res = fit_arima(series, order=(1, 0, 0), forecast_steps=3)
    assert res.forecast.shape[0] == 3


def test_fit_exponential_smoothing() -> None:
    series = np.sin(np.linspace(0, 2 * np.pi, 30))
    res = fit_exponential_smoothing(series, trend=None, seasonal=None, forecast_steps=3)
    assert res.forecast.shape[0] == 3
