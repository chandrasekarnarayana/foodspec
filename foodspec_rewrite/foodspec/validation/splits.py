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

Deterministic group-aware splitters for validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Iterator, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold


@dataclass
class LeaveOneGroupOutSplitter:
    """Leave-One-Group-Out splitter.

    Yields (train_idx, test_idx) pairs; one group held out each fold.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.zeros((6, 2))
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> groups = np.array([1, 1, 2, 2, 3, 3])
    >>> splitter = LeaveOneGroupOutSplitter()
    >>> folds = list(splitter.split(X, y, groups))
    >>> len(folds)
    3
    >>> sorted(len(te) for _, te in folds)
    [2, 2, 2]
    """

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: Sequence[object]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)
        if X.shape[0] != y.shape[0] or y.shape[0] != groups.shape[0]:
            raise ValueError("X, y, and groups must have the same length")
        logo = LeaveOneGroupOut()
        for tr, te in logo.split(X, y, groups=groups):
            yield tr, te


@dataclass
class StratifiedKFoldOrGroupKFold:
    """Stratified K-Fold with group fallback and deterministic shuffling.

    Behavior:
    - If `groups` are provided and at least `n_splits` unique groups exist,
      use `GroupKFold(n_splits)`. This avoids leakage across groups.
    - Otherwise, use `StratifiedKFold(n_splits, shuffle=True, random_state=seed)`.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    seed : int, default 0
        Random seed for deterministic stratified shuffling.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.zeros((10, 2))
    >>> y = np.array([0, 1] * 5)
    >>> g = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    >>> splitter = StratifiedKFoldOrGroupKFold(n_splits=5, seed=42)
    >>> # With sufficient groups -> GroupKFold used
    >>> folds = list(splitter.split(X, y, g))
    >>> len(folds)
    5
    >>> # Without groups -> StratifiedKFold used and deterministic
    >>> folds2 = list(splitter.split(X, y))
    >>> len(folds2)
    5
    """

    n_splits: int = 5
    seed: int = 0

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: Sequence[object] | None = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape[0] != X.shape[0]:
                raise ValueError("groups must have the same length as X")
            unique_groups = np.unique(groups)
            if unique_groups.size >= self.n_splits:
                gkf = GroupKFold(n_splits=self.n_splits)
                for tr, te in gkf.split(X, y, groups=groups):
                    yield tr, te
                return
        # Fallback to stratified with deterministic shuffle
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for tr, te in skf.split(X, y):
            yield tr, te


__all__ = ["LeaveOneGroupOutSplitter", "StratifiedKFoldOrGroupKFold"]
