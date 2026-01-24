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
I/O module: Data loading, format detection, library management.

Loading spectral data from CSV:
    from foodspec.io import load_csv_spectra
    dataset = load_csv_spectra("./data/oils.csv", data_spec)
"""
from foodspec.io.readers import load_csv_spectra

__all__ = ["load_csv_spectra"]
