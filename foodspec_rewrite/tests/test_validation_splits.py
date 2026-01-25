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

from foodspec.validation import LeaveOneGroupOutSplitter, StratifiedKFoldOrGroupKFold


def test_leave_one_group_out_yields_each_group_once() -> None:
    X = np.zeros((6, 2))
    y = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array([1, 1, 2, 2, 3, 3])

    splitter = LeaveOneGroupOutSplitter()
    folds = list(splitter.split(X, y, groups))

    assert len(folds) == 3
    # Each test fold contains exactly one group's indices
    test_groups = []
    for _, te in folds:
        assert len(te) == 2
        test_groups.append(groups[te][0])
        assert np.all(groups[te] == groups[te][0])
    assert set(test_groups) == {1, 2, 3}


def test_stratified_fallback_is_deterministic_without_groups() -> None:
    X = np.zeros((20, 3))
    y = np.array([0, 1] * 10)

    s1 = StratifiedKFoldOrGroupKFold(n_splits=5, seed=123)
    s2 = StratifiedKFoldOrGroupKFold(n_splits=5, seed=123)

    folds1 = list(s1.split(X, y))
    folds2 = list(s2.split(X, y))

    # Deterministic across instances with same seed
    for (tr1, te1), (tr2, te2) in zip(folds1, folds2):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


def test_groupkfold_used_when_enough_groups() -> None:
    X = np.zeros((10, 2))
    y = np.array([0, 1] * 5)
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    s = StratifiedKFoldOrGroupKFold(n_splits=5, seed=0)
    folds = list(s.split(X, y, groups))

    # Group-wise separation: no group should appear in both train and test for a fold
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))
    assert len(folds) == 5


def test_splitter_input_validation() -> None:
    X = np.zeros((4, 2))
    y = np.array([0, 1, 0, 1])
    groups = np.array([1, 1, 2, 2])

    with pytest.raises(ValueError):
        list(LeaveOneGroupOutSplitter().split(X, y, groups[:2]))

    with pytest.raises(ValueError):
        list(StratifiedKFoldOrGroupKFold(n_splits=3).split(X[:2], y))

    with pytest.raises(ValueError):
        list(StratifiedKFoldOrGroupKFold(n_splits=3).split(X, y, groups[:2]))
