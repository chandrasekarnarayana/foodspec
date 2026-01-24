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

Extracting features from spectra:
    from foodspec.features import PeakRatios
    ratios = PeakRatios(peak_pairs=[(1030, 1050)], window=15).compute(X, x)
"""
from foodspec.features.peaks import PeakRatios
from foodspec.features.chemometrics import PCAFeatureExtractor

__all__ = ["PeakRatios", "PCAFeatureExtractor"]
