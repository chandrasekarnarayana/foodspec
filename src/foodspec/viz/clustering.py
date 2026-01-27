from __future__ import annotations

"""Clustering visualization helpers."""

from typing import Iterable, Optional

import matplotlib.pyplot as plt

from foodspec.viz.style import apply_style

try:  # optional
    from scipy.cluster.hierarchy import dendrogram
except Exception:  # pragma: no cover
    dendrogram = None


def plot_dendrogram(linkage_matrix, *, labels: Optional[Iterable[str]] = None, title: str = "Dendrogram") -> plt.Figure:
    """Plot hierarchical clustering dendrogram."""
    if dendrogram is None:  # pragma: no cover
        raise ImportError("scipy is required for dendrogram plotting.")
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    dendrogram(linkage_matrix, labels=list(labels) if labels is not None else None, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig


__all__ = ["plot_dendrogram"]
