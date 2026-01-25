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

Deterministic group-aware splitters for validation with metadata support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold


@dataclass
class LeaveOneGroupOutSplitter:
    """Leave-One-Group-Out splitter with metadata support and backward compatibility.

    Supports both legacy API (with groups array) and new metadata-based API.
    Yields (train_idx, test_idx) pairs or (train_idx, test_idx, fold_info) tuples
    depending on input format. Groups are ordered deterministically.

    Parameters
    ----------
    group_key : str, default "group"
        Column name in metadata DataFrame for group identifiers.
        Only used when meta is a DataFrame; ignored for legacy groups array.

    Examples
    --------
    Legacy API (groups as array):
    
    >>> import numpy as np
    >>> X = np.zeros((6, 2))
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> groups = np.array([1, 1, 2, 2, 3, 3])
    >>> splitter = LeaveOneGroupOutSplitter()
    >>> folds = list(splitter.split(X, y, groups))
    >>> len(folds)
    3

    New API (metadata DataFrame):
    
    >>> import pandas as pd
    >>> meta = pd.DataFrame({"group": [1, 1, 2, 2, 3, 3]})
    >>> folds = list(splitter.split(X, y, meta))
    >>> len(folds)
    3
    >>> train_idx, test_idx, fold_info = folds[0]
    >>> fold_info
    {'fold_id': 0, 'held_out_group': 1, 'n_train': 4, 'n_test': 2}
    """

    group_key: str = "group"

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups_or_meta: np.ndarray | pd.DataFrame | Sequence[object],
    ):
        """Split data leaving one group out at a time.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : ndarray, shape (n_samples,)
            Target labels.
        groups_or_meta : ndarray, DataFrame, or sequence
            Either:
            - Legacy API: array-like of group identifiers
            - New API: DataFrame with group column

        Yields
        ------
        Legacy API (array input):
            train_idx, test_idx : tuple of ndarrays
        
        New API (DataFrame input):
            train_idx, test_idx, fold_info : tuple with dict

        Raises
        ------
        ValueError
            If group_key not in metadata columns (DataFrame only).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Detect which API is being used
        if isinstance(groups_or_meta, pd.DataFrame):
            # New API: metadata-based
            yield from self._split_with_metadata(X, y, groups_or_meta)
        else:
            # Legacy API: groups array
            yield from self._split_with_groups_array(X, y, groups_or_meta)

    def _split_with_metadata(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """New API: Split using metadata DataFrame.
        
        Yields (train_idx, test_idx, fold_info).
        """
        if X.shape[0] != y.shape[0] or y.shape[0] != len(meta):
            raise ValueError("X, y, and meta must have the same length")

        # Validate group_key exists
        if self.group_key not in meta.columns:
            available = list(meta.columns)
            raise ValueError(
                f"Group key '{self.group_key}' not found in metadata. "
                f"Available keys: {available}"
            )

        groups = np.asarray(meta[self.group_key])

        # Get unique groups and sort deterministically
        unique_groups = np.unique(groups)
        unique_groups = np.sort(unique_groups)

        fold_id = 0
        for held_out_group in unique_groups:
            test_mask = groups == held_out_group
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]

            fold_info = {
                "fold_id": fold_id,
                "held_out_group": held_out_group,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }

            yield train_idx, test_idx, fold_info
            fold_id += 1

    def _split_with_groups_array(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Sequence[object],
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Legacy API: Split using groups array.
        
        Yields (train_idx, test_idx) pairs.
        """
        groups = np.asarray(groups)
        if X.shape[0] != y.shape[0] or y.shape[0] != groups.shape[0]:
            raise ValueError("X, y, and groups must have the same length")

        # Get unique groups and sort deterministically
        unique_groups = np.unique(groups)
        unique_groups = np.sort(unique_groups)

        for held_out_group in unique_groups:
            test_mask = groups == held_out_group
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]

            yield train_idx, test_idx


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


@dataclass
class LeaveOneBatchOutSplitter:
    """Convenience wrapper for Leave-One-Batch-Out using "batch" metadata key.

    Equivalent to LeaveOneGroupOutSplitter(group_key="batch").

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.zeros((6, 2))
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> meta = pd.DataFrame({
    ...     "batch": ["A", "A", "B", "B", "C", "C"],
    ...     "date": ["2024-01-01"] * 6,
    ... })
    >>> splitter = LeaveOneBatchOutSplitter()
    >>> folds = list(splitter.split(X, y, meta))
    >>> len(folds)
    3
    >>> train_idx, test_idx, fold_info = folds[0]
    >>> fold_info["held_out_group"]
    'A'
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Split data leaving one batch out at a time.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : ndarray, shape (n_samples,)
            Target labels.
        meta : DataFrame
            Metadata with "batch" column.

        Yields
        ------
        train_idx, test_idx, fold_info
            Training indices, test indices, and fold metadata.

        Raises
        ------
        ValueError
            If "batch" column not in metadata.
        """
        splitter = LeaveOneGroupOutSplitter(group_key="batch")
        yield from splitter.split(X, y, meta)


@dataclass
class LeaveOneStageOutSplitter:
    """Convenience wrapper for Leave-One-Stage-Out using "stage" metadata key.

    Equivalent to LeaveOneGroupOutSplitter(group_key="stage").

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.zeros((6, 2))
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> meta = pd.DataFrame({
    ...     "stage": ["discovery", "discovery", "validation", "validation", "test", "test"],
    ... })
    >>> splitter = LeaveOneStageOutSplitter()
    >>> folds = list(splitter.split(X, y, meta))
    >>> len(folds)
    3
    >>> train_idx, test_idx, fold_info = folds[0]
    >>> fold_info["held_out_group"]
    'discovery'
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Split data leaving one stage out at a time.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : ndarray, shape (n_samples,)
            Target labels.
        meta : DataFrame
            Metadata with "stage" column.

        Yields
        ------
        train_idx, test_idx, fold_info
            Training indices, test indices, and fold metadata.

        Raises
        ------
        ValueError
            If "stage" column not in metadata.
        """
        splitter = LeaveOneGroupOutSplitter(group_key="stage")
        yield from splitter.split(X, y, meta)


__all__ = [
    "LeaveOneGroupOutSplitter",
    "LeaveOneBatchOutSplitter",
    "LeaveOneStageOutSplitter",
    "StratifiedKFoldOrGroupKFold",
]
