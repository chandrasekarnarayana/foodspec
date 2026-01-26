from __future__ import annotations
"""Control chart plotting helpers."""

from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from foodspec.qc.control_charts import ControlChartGroup, ControlChartResult
from foodspec.viz.style import apply_style


def _plot_chart(ax, result: ControlChartResult, *, title: str, ylabel: str) -> None:
    ax.plot(result.points, marker="o", linestyle="-", color="#2a6fdb")
    ax.axhline(result.center, color="black", linestyle="--", linewidth=1)
    ax.axhline(result.ucl, color="red", linestyle=":", linewidth=1)
    ax.axhline(result.lcl, color="red", linestyle=":", linewidth=1)
    if result.signals:
        ax.scatter(result.signals, result.points[result.signals], color="red", zorder=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Index")


def plot_control_chart(result: ControlChartResult, *, title: Optional[str] = None) -> plt.Figure:
    """Plot a single control chart."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    _plot_chart(ax, result, title=title or result.chart.upper(), ylabel=result.chart)
    fig.tight_layout()
    return fig


def plot_control_chart_group(group: ControlChartGroup, *, title: str = "Control Chart") -> plt.Figure:
    """Plot paired charts (X-bar + variability or Individuals + MR)."""
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
    _plot_chart(axes[0], group.xbar, title=f"{title} - {group.xbar.chart}", ylabel=group.xbar.chart)
    _plot_chart(
        axes[1],
        group.variability,
        title=f"{title} - {group.variability.chart}",
        ylabel=group.variability.chart,
    )
    fig.tight_layout()
    return fig


def plot_cusum(pos: np.ndarray, neg: np.ndarray, h: float, *, title: str = "CUSUM") -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(pos, label="CUSUM +", color="#2a6fdb")
    ax.plot(neg, label="CUSUM -", color="#f39c12")
    ax.axhline(h, color="red", linestyle=":")
    ax.axhline(-h, color="red", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("CUSUM")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ewma(ewma: np.ndarray, lcl: np.ndarray, ucl: np.ndarray, *, title: str = "EWMA") -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ewma, label="EWMA", color="#2a6fdb")
    ax.plot(lcl, color="red", linestyle=":", label="LCL/UCL")
    ax.plot(ucl, color="red", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("EWMA")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pareto(counts: Dict[str, int], *, title: str = "Pareto Chart") -> plt.Figure:
    apply_style()
    labels = list(counts.keys())
    values = np.asarray(list(counts.values()), dtype=float)
    cum = np.cumsum(values) / np.sum(values) if values.size else values
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, values, color="#2a6fdb")
    ax2 = ax.twinx()
    ax2.plot(labels, cum * 100, color="#f39c12", marker="o")
    ax2.set_ylabel("Cumulative %")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_runs(values: Iterable[float], *, title: str = "Runs Analysis") -> plt.Figure:
    apply_style()
    vals = np.asarray(list(values), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(vals, marker="o", linestyle="-", color="#2a6fdb")
    ax.axhline(np.median(vals), color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_control_chart",
    "plot_control_chart_group",
    "plot_cusum",
    "plot_ewma",
    "plot_pareto",
    "plot_runs",
]
