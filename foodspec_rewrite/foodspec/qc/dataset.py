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

Dataset-level QC metrics for batches and replicates.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


class DatasetQC:
    """Dataset-level QC for batch drift and replicate consistency.

    Parameters
    ----------
    batch_col : str, default "batch"
        Metadata column used to group samples into batches.
    replicate_col : str, default "replicate_id"
        Metadata column used to group technical replicates.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = np.array([[1.0, 1.0], [1.1, 0.9], [2.0, 2.0]])
    >>> meta = pd.DataFrame({"batch": ["A", "A", "B"], "replicate_id": ["r1", "r1", "r2"]})
    >>> qc = DatasetQC()
    >>> qc.compute_batch_drift(X, meta)["drift_score"].tolist()
    [0.0, 1.0035978264796036]
    >>> qc.compute_replicate_consistency(X, meta)["within_variance"].tolist()
    [0.0024999999999999823, 0.0]
    """

    def __init__(self, batch_col: str = "batch", replicate_col: str = "replicate_id") -> None:
        self.batch_col = batch_col
        self.replicate_col = replicate_col

    @staticmethod
    def _validate_inputs(X: np.ndarray, meta: pd.DataFrame) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        if len(meta) != X.shape[0]:
            raise ValueError("metadata rows must align with X rows")

    @staticmethod
    def _group_indices(meta: pd.DataFrame, column: str) -> Iterable[Tuple[object, np.ndarray]]:
        for value, frame in meta.groupby(column, sort=False):
            yield value, frame.index.to_numpy()

    def compute_batch_drift(self, X: np.ndarray, meta: pd.DataFrame, reference: str | None = "first") -> pd.DataFrame:
        """Compute per-batch drift scores relative to a reference batch or global mean.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_wavenumbers)
            Spectral matrix.
        meta : DataFrame
            Per-sample metadata containing a batch column.
        reference : {"first", "global", str, None}, default "first"
            How to choose the reference mean spectrum.
            - "first": mean of the first batch encountered.
            - "global" or None: mean of all samples.
            - str: mean of the named batch.

        Returns
        -------
        DataFrame
            Columns: [batch_col, "n_samples", "reference", "drift_score"].
        """

        meta = meta.reset_index(drop=True)
        self._validate_inputs(X, meta)
        if self.batch_col not in meta.columns:
            raise ValueError(f"Metadata missing batch column '{self.batch_col}'")

        groups = list(self._group_indices(meta, self.batch_col))
        if not groups:
            return pd.DataFrame(columns=[self.batch_col, "n_samples", "reference", "drift_score"])

        if reference in (None, "global"):
            ref_mean = X.mean(axis=0)
            ref_label = "global"
        elif reference == "first":
            first_indices = groups[0][1]
            ref_mean = X[first_indices].mean(axis=0)
            ref_label = str(groups[0][0])
        else:
            if reference not in meta[self.batch_col].unique():
                raise ValueError(f"Reference batch '{reference}' not found in metadata")
            ref_mean = X[meta[self.batch_col] == reference].mean(axis=0)
            ref_label = str(reference)

        norm = float(np.sqrt(X.shape[1]))
        rows: list[Dict[str, object]] = []
        for batch_value, indices in groups:
            batch_X = X[indices]
            batch_mean = batch_X.mean(axis=0)
            drift = float(np.linalg.norm(batch_mean - ref_mean) / norm)
            rows.append(
                {
                    self.batch_col: batch_value,
                    "n_samples": int(batch_X.shape[0]),
                    "reference": ref_label,
                    "drift_score": drift,
                }
            )

        return pd.DataFrame(rows)

    def compute_replicate_consistency(self, X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        """Compute within-replicate variance per group.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_wavenumbers)
            Spectral matrix.
        meta : DataFrame
            Per-sample metadata containing a replicate column.

        Returns
        -------
        DataFrame
            Columns: [replicate_col, "n_samples", "within_variance"].
        """

        meta = meta.reset_index(drop=True)
        self._validate_inputs(X, meta)
        if self.replicate_col not in meta.columns:
            raise ValueError(f"Metadata missing replicate column '{self.replicate_col}'")

        rows: list[Dict[str, object]] = []
        for replicate_value, indices in self._group_indices(meta, self.replicate_col):
            replicate_X = X[indices]
            if replicate_X.shape[0] <= 1:
                variance = 0.0
            else:
                variance = float(np.var(replicate_X, axis=0, ddof=0).mean())
            rows.append(
                {
                    self.replicate_col: replicate_value,
                    "n_samples": int(replicate_X.shape[0]),
                    "within_variance": variance,
                }
            )

        return pd.DataFrame(rows)

    def compute(self, X: np.ndarray, meta: pd.DataFrame, reference: str | None = "first") -> Dict[str, pd.DataFrame]:
        """Compute both batch drift and replicate consistency tables.

        Returns
        -------
        dict
            Keys: "batch_drift", "replicate_consistency".
        """

        return {
            "batch_drift": self.compute_batch_drift(X, meta, reference=reference),
            "replicate_consistency": self.compute_replicate_consistency(X, meta),
        }


__all__ = ["DatasetQC"]
