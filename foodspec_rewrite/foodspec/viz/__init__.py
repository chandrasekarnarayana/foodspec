"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
Visualization module: Spectral plots and diagnostics.

Visualizing spectral data and model results:
    from foodspec.viz import plot_raw_vs_processed, plot_pca_scatter
    plot_raw_vs_processed(wavenumbers, raw, processed, artifacts)
    plot_pca_scatter(scores, labels, artifacts)
"""

from foodspec.viz.plots import (
    plot_confusion_matrix,
    plot_heatmap,
    plot_pca_scatter,
    plot_raw_vs_processed,
    plot_reliability_diagram,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_heatmap",
    "plot_pca_scatter",
    "plot_raw_vs_processed",
    "plot_reliability_diagram",
]
