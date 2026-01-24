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

from foodspec.validation import accuracy, macro_f1, auroc_ovr, ece, brier


def test_accuracy_and_macro_f1_from_probabilities() -> None:
    # y: two correct, two incorrect with argmax
    y = np.array([0, 1, 1, 0])
    proba = np.array([
        [0.9, 0.1],  # correct (0)
        [0.4, 0.6],  # correct (1)
        [0.7, 0.3],  # incorrect (pred 0)
        [0.2, 0.8],  # incorrect (pred 1)
    ])

    acc = accuracy(y, proba)
    f1 = macro_f1(y, proba)

    assert acc == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_auroc_ovr_perfect_separation_binary_and_multiclass() -> None:
    # Binary perfect separation -> AUROC = 1.0
    y_bin = np.array([0, 0, 1, 1])
    proba_bin = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    assert auroc_ovr(y_bin, proba_bin) == pytest.approx(1.0)

    # Multiclass perfect separation (ovr macro)
    y_mc = np.array([0, 1, 2])
    proba_mc = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])
    assert auroc_ovr(y_mc, proba_mc) == pytest.approx(1.0)


def test_ece_basic_and_extreme_cases() -> None:
    # Perfect predictions with confidence 1.0 -> ECE = 0
    y = np.array([0, 1, 0, 1])
    proba_perfect = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    assert ece(y, proba_perfect, n_bins=5) == pytest.approx(0.0)

    # Constant high confidence for class 0; half wrong -> ECE ~ |0.5 - 0.9| = 0.4
    y = np.array([0, 1, 1, 0])
    proba_bad = np.array([[0.9, 0.1]] * 4)
    assert ece(y, proba_bad, n_bins=10) == pytest.approx(0.4)


def test_brier_multiclass_and_binary_vector() -> None:
    # Binary 2-column probabilities
    y = np.array([0, 1])
    proba = np.array([[0.8, 0.2], [0.7, 0.3]])
    # Expected = ((0.2^2 + 0.2^2) + (0.7^2 + 0.7^2)) / 2 = (0.08 + 0.98)/2 = 0.53
    assert brier(y, proba) == pytest.approx(0.53)

    # Binary vector probabilities (positive class only)
    y = np.array([0, 1, 1])
    proba_vec = np.array([0.1, 0.8, 0.9])
    val = brier(y, proba_vec)  # should not raise and be in [0, 2]
    assert 0.0 <= val <= 2.0
