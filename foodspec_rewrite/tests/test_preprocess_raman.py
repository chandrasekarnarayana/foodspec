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

from foodspec.preprocess.raman import DespikeHampel, AsLSBaseline


def test_despike_hampel_reduces_spike() -> None:
    X = np.array([[1.0, 1.0, 10.0, 1.0, 1.0]])
    filt = DespikeHampel(window_size=1, threshold=3.0)
    out = filt.transform(X)
    assert out.shape == X.shape
    # Spike replaced by neighborhood median (1.0)
    assert np.isclose(out[0, 2], 1.0)


def test_asls_baseline_reduces_drift() -> None:
    x = np.linspace(0, 1, 50)
    drift = 0.5 * x  # slow drift
    signal = np.exp(-((x - 0.5) ** 2) / 0.01)
    y = drift + signal
    X = y.reshape(1, -1)

    asls = AsLSBaseline(lam=1e4, p=0.01, n_iter=10)
    corrected = asls.transform(X)
    assert corrected.shape == X.shape
    # Drift should be largely removed; median near zero
    assert abs(np.median(corrected)) < 0.1
