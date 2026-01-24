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
import pandas as pd

from foodspec.qc.spectral import SpectralQC


def test_spectral_qc_snr_and_clipping() -> None:
    clean = np.linspace(0, 1, 50)
    noisy = clean + np.random.normal(0, 0.5, size=clean.shape)
    clipped = np.clip(clean.copy(), 0.2, 0.8)

    X = np.vstack([clean, noisy, clipped])
    meta = pd.DataFrame(index=[0, 1, 2])

    qc = SpectralQC()
    df = qc.compute(X, meta)

    assert set(df.columns) == {"snr", "clip_frac", "entropy"}
    assert df.shape[0] == 3
    assert df.loc[1, "snr"] < df.loc[0, "snr"]  # noisy should have lower SNR
    assert df.loc[2, "clip_frac"] > df.loc[0, "clip_frac"]  # clipped higher
