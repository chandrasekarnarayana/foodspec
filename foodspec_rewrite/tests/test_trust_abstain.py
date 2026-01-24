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

from foodspec.trust import evaluate_abstention


def test_abstain_on_low_confidence_only() -> None:
    proba = np.array([
        [0.85, 0.15],  # confident
        [0.55, 0.45],  # low conf -> abstain
        [0.9, 0.1],    # confident
    ])
    y_true = np.array([0, 1, 0])

    res = evaluate_abstention(proba, y_true, threshold=0.8)

    assert res.abstain_mask == [False, True, False]
    assert res.predictions == [0, 0, 0]
    assert res.abstain_rate == pytest.approx(1 / 3)
    assert res.accuracy_non_abstained == pytest.approx(1.0)
    assert res.coverage is None


def test_abstain_on_large_prediction_sets_and_report_coverage() -> None:
    proba = np.array([
        [0.6, 0.3, 0.1],  # set size 2 -> abstain because >1
        [0.7, 0.2, 0.1],  # set size 1 -> keep
        [0.4, 0.3, 0.3],  # low conf -> abstain regardless
    ])
    y_true = np.array([1, 0, 2])
    prediction_sets = [[0, 1], [0], [0, 1, 2]]

    res = evaluate_abstention(proba, y_true, threshold=0.5, prediction_sets=prediction_sets, max_set_size=1)

    assert res.abstain_mask == [True, False, True]
    assert res.set_sizes == [2, 1, 3]
    assert res.abstain_rate == pytest.approx(2 / 3)
    # Only second sample kept: predicted 0, true 0 -> accuracy 1.0
    assert res.accuracy_non_abstained == pytest.approx(1.0)
    # Coverage across all samples: sets contain true labels for 1st (yes), 2nd (yes), 3rd (yes)
    assert res.coverage == pytest.approx(1.0)


def test_all_abstained_returns_none_accuracy() -> None:
    proba = np.array([[0.4, 0.6], [0.3, 0.7]])
    y_true = np.array([1, 1])

    res = evaluate_abstention(proba, y_true, threshold=0.9)

    assert res.abstain_mask == [True, True]
    assert res.accuracy_non_abstained is None
    assert res.abstain_rate == pytest.approx(1.0)


def test_validation_errors() -> None:
    proba = np.array([[0.6, 0.4]])
    y_true = np.array([0])
    with pytest.raises(ValueError):
        evaluate_abstention(proba, y_true, threshold=1.1)
    with pytest.raises(ValueError):
        evaluate_abstention(proba[0], y_true, threshold=0.5)
    with pytest.raises(ValueError):
        evaluate_abstention(proba, y_true, threshold=0.5, prediction_sets=[[0], [1]], max_set_size=1)
    with pytest.raises(ValueError):
        evaluate_abstention(proba, y_true, threshold=0.5, prediction_sets=[[0]], max_set_size=0)
