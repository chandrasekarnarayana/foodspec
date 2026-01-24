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
import pytest

from foodspec.preprocess.common import SavitzkyGolay, SNV, VectorNormalize, Derivative


def test_savitzky_golay_smoothing() -> None:
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    sg = SavitzkyGolay(window_length=3, polyorder=1)
    out = sg.transform(X)
    assert out.shape == X.shape


def test_snv_normalization() -> None:
    X = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    snv = SNV()
    out = snv.transform(X)
    assert np.allclose(out[0].mean(), 0.0)
    with pytest.raises(ValueError):
        snv.transform(np.array([[0.0, 0.0, 0.0]]))


def test_vector_normalize() -> None:
    X = np.array([[3.0, 4.0]])
    vn = VectorNormalize()
    out = vn.transform(X)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    with pytest.raises(ValueError):
        vn.transform(np.array([[0.0, 0.0]]))


def test_derivative_first_and_second() -> None:
    X = np.array([[1.0, 2.0, 4.0, 7.0]])
    d1 = Derivative(order=1)
    out1 = d1.transform(X)
    assert np.allclose(out1, np.diff(X, axis=1))

    d2 = Derivative(order=2)
    out2 = d2.transform(X)
    assert np.allclose(out2, np.diff(X, n=2, axis=1))

    with pytest.raises(ValueError):
        Derivative(order=3)
