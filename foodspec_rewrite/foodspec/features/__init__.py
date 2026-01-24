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

Features module: Spectral, statistical, and domain-specific feature extraction.

Usage Examples
--------------

Peak-based features::

    from foodspec.features import PeakRatios, PeakHeights, PeakAreas
    heights = PeakHeights([1030, 1050], window=15).compute(X, x)
    areas = PeakAreas([1030], window=20, baseline_subtract=True).compute(X, x)
    ratios = PeakRatios([(1030, 1050)], window=15).compute(X, x)

Band integration::

    from foodspec.features import BandIntegration
    bands = BandIntegration([(1400, 1600), (2800, 3000)]).compute(X, x)

Chemometric embeddings::

    from foodspec.features import PCAFeatureExtractor, PLSFeatureExtractor
    pca = PCAFeatureExtractor(n_components=5)
    pca.fit(X_train)
    features_train = pca.transform(X_train)
    features_test = pca.transform(X_test)
    
    pls = PLSFeatureExtractor(n_components=3)
    pls.fit(X_train, y_train)  # Supervised
    features_test = pls.transform(X_test)

Hybrid feature composition::

    from foodspec.features import FeatureComposer
    composer = FeatureComposer([
        ("pca", PCAFeatureExtractor(n_components=3), {}),
        ("peaks", PeakHeights([1200, 1500]), {"x": x_grid}),
        ("bands", BandIntegration([(1400, 1600)]), {"x": x_grid}),
    ])
    composer.fit(X_train, x=x_grid)
    feature_set = composer.transform(X_test, x=x_grid)

Feature selection for marker panels::

    from foodspec.features import StabilitySelector
    selector = StabilitySelector(
        estimator_factory=lambda: SparseClassifier(),
        n_resamples=50,
        selection_threshold=0.5,
    )
    selector.fit(X_train, y_train)
    X_selected = selector.transform(X_test)
    marker_panel = selector.get_marker_panel(x_wavenumbers=x_grid)
"""

from foodspec.features.base import FeatureExtractor, FeatureSet
from foodspec.features.bands import BandIntegration
from foodspec.features.chemometrics import PCAFeatureExtractor, PLSFeatureExtractor
from foodspec.features.composer import FeatureComposer
from foodspec.features.hybrid import FeatureUnion
from foodspec.features.marker_panel import MarkerPanel
from foodspec.features.peaks import PeakAreas, PeakHeights, PeakRatios
from foodspec.features.selection import StabilitySelector

__all__ = [
    # Protocols and base classes
    "FeatureExtractor",
    "FeatureSet",
    # Peak-based features
    "PeakRatios",
    "PeakHeights",
    "PeakAreas",
    # Band integration
    "BandIntegration",
    # Chemometric embeddings
    "PCAFeatureExtractor",
    "PLSFeatureExtractor",
    # Hybrid composition
    "FeatureComposer",
    "FeatureUnion",
    "MarkerPanel",
    # Feature selection
    "StabilitySelector",
]
