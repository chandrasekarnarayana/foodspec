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
import pytest

from foodspec.features import PeakRatios


def _gauss(x: np.ndarray, mu: float, amp: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def test_peak_ratios_basic_behavior() -> None:
    x = np.linspace(1000, 1100, 101)  # 1 unit step

    # Sample 1: stronger at 1030 than 1050
    y1 = _gauss(x, 1030, amp=2.0) + _gauss(x, 1050, amp=1.0)
    # Sample 2: flipped strengths
    y2 = _gauss(x, 1030, amp=1.0) + _gauss(x, 1050, amp=2.0)

    X = np.vstack([y1, y2])

    pr = PeakRatios(pairs=[(1030, 1050)], window=10.0)
    df = pr.compute(X, x)

    assert list(df.columns) == ["ratio@1030/1050"]
    assert df.shape == (2, 1)
    assert df.loc[0, "ratio@1030/1050"] > 1.0  # sample 1 has higher 1030 peak
    assert df.loc[1, "ratio@1030/1050"] < 1.0  # sample 2 has higher 1050 peak


def test_peak_ratios_window_captures_shifted_peaks() -> None:
    x = np.linspace(1000, 1100, 101)

    # Shift actual peaks by +2 units away from targets
    y = _gauss(x, 1032, amp=1.5) + _gauss(x, 1052, amp=0.5)
    X = np.vstack([y])

    pr = PeakRatios(pairs=[(1030, 1050)], window=5.0)
    df = pr.compute(X, x)

    # Despite target mismatch, window search should find the shifted maxima
    assert df.loc[0, "ratio@1030/1050"] > 1.0


def test_peak_ratios_input_validation() -> None:
    x = np.linspace(100, 200, 10)
    X = np.ones((2, 10))

    pr = PeakRatios(pairs=[(120, 150)])

    with pytest.raises(ValueError):
        pr.compute(X.reshape(20, 1), x)  # wrong X shape

    with pytest.raises(ValueError):
        pr.compute(X, x[:5])  # mismatched x

    with pytest.raises(ValueError):
        PeakRatios(pairs=[])  # empty pairs - should fail at initialization
