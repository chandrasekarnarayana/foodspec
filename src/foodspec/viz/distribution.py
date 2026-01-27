from __future__ import annotations

"""Distribution diagnostic plots."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from foodspec.stats.distribution_fitting import probability_plot_data
from foodspec.viz.style import apply_style


def plot_probability_plot(
    data: np.ndarray,
    *,
    dist: str = "normal",
    title: Optional[str] = None,
) -> plt.Figure:
    """Probability plot for distribution diagnostics."""
    apply_style()
    info = probability_plot_data(data, dist_name=dist)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(info["theoretical"], info["ordered"], color="#2a6fdb", alpha=0.7)
    line_x = np.linspace(np.min(info["theoretical"]), np.max(info["theoretical"]), 100)
    line_y = info["slope"] * line_x + info["intercept"]
    ax.plot(line_x, line_y, color="red", linestyle="--")
    ax.set_title(title or f"{dist.title()} Probability Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Ordered Values")
    fig.tight_layout()
    return fig


__all__ = ["plot_probability_plot"]
