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
Visualization module: Spectral plots, model diagnostics, interactive dashboards.

Visualizing spectral data and model results:
    from foodspec.viz import plot_spectra, plot_pca
    plot_spectra(spectra, wavenumbers)
    plot_pca(X_features, labels=y)
"""

__all__ = []
