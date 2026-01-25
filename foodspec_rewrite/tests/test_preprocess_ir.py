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
"""

import numpy as np

from foodspec.preprocess.ir import (
    MultiplicativeScatterCorrection,
    ExtendedMultiplicativeScatterCorrection,
)


def test_msc_removes_scatter() -> None:
    ref = np.array([1.0, 2.0, 3.0])
    X = np.vstack([ref, ref * 2 + 1.0])
    msc = MultiplicativeScatterCorrection().fit(X)
    out = msc.transform(X)
    assert out.shape == X.shape
    # Second spectrum should align to reference after correction
    assert np.allclose(out[1], out[0], atol=1e-6)


def test_emsc_baseline_and_scale() -> None:
    x_axis = np.linspace(0, 1, 5)
    ref = np.array([1, 2, 3, 4, 5], dtype=float)
    baseline = 0.5 + 0.2 * x_axis
    scaled = 2.0 * ref + baseline
    X = np.vstack([ref, scaled])

    emsc = ExtendedMultiplicativeScatterCorrection(degree=1).fit(X)
    out = emsc.transform(X)
    assert out.shape == X.shape
    # Baseline and scaling removed for second spectrum
    assert np.allclose(out[1], out[0], atol=1e-6)
