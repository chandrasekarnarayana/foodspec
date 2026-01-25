"""Visualization helpers for heating degradation analysis."""
from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

__all__ = ["plot_ratio_vs_time"]


def plot_ratio_vs_time(time, ratio, model: LinearRegression | None = None, ax=None, label: str | None = None):
    """Plot ratio vs time with optional fitted trend line."""

    ax = ax or plt.gca()
    time_arr = np.asarray(time, dtype=float)
    ratio_arr = np.asarray(ratio, dtype=float)
    ax.scatter(time_arr, ratio_arr, alpha=0.7, label=label or "data")
    if model is not None and hasattr(model, "predict"):
        t_line = np.linspace(time_arr.min(), time_arr.max(), 50).reshape(-1, 1)
        r_line = model.predict(t_line)
        ax.plot(t_line, r_line, color="C1", label="trend")
    ax.set_xlabel("Time")
    ax.set_ylabel("Ratio")
    if label or model is not None:
        ax.legend()
    return ax
