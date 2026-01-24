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

Data structures for spectra and datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _ensure_numeric(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype {array.dtype}")


def _ensure_no_nan(array: np.ndarray, name: str) -> None:
    if np.isnan(array).any():
        raise ValueError(f"{name} contains NaN values; clean or allow explicitly")


@dataclass
class Spectrum:
    """Single spectrum (x grid, y intensities, metadata).

    Parameters
    ----------
    x : array-like
        Wavenumber/wavelength grid.
    y : array-like
        Intensities aligned to ``x``.
    metadata : dict
        Arbitrary metadata (sample_id, modality, etc.).
    allow_nans : bool, default False
        If False, NaNs are rejected.

    Examples
    --------
    >>> s = Spectrum(x=[1, 2, 3], y=[10, 20, 30], metadata={"sample_id": "A"})
    >>> s.y.mean()
    20.0
    """

    x: np.ndarray
    y: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    allow_nans: bool = False

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape")
        _ensure_numeric(self.x, "x")
        _ensure_numeric(self.y, "y")
        if not self.allow_nans:
            _ensure_no_nan(self.x, "x")
            _ensure_no_nan(self.y, "y")


@dataclass
class SpectraSet:
    """Collection of aligned spectra with metadata.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_wavenumbers)
        Intensity matrix.
    x : array-like, shape (n_wavenumbers,)
        Shared wavenumber grid.
    y : array-like, optional
        Labels per sample.
    metadata : DataFrame, optional
        Per-sample metadata; must align on rows.
    allow_nans : bool, default False
        If False, NaNs are rejected.

    Examples
    --------
    >>> import pandas as pd
    >>> X = [[1, 2], [3, 4]]
    >>> x = [100, 200]
    >>> meta = pd.DataFrame({"sample_id": ["a", "b"]})
    >>> ds = SpectraSet(X=X, x=x, y=[0, 1], metadata=meta)
    >>> ds.summary_stats()["mean"].tolist()
    [2.0, 3.0]
    """

    X: np.ndarray
    x: np.ndarray
    y: Optional[np.ndarray] = None
    metadata: Optional[pd.DataFrame] = None
    allow_nans: bool = False

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        n_samples, n_wavenumbers = self.X.shape
        if self.x.shape != (n_wavenumbers,):
            raise ValueError("x must have length matching X.shape[1]")
        _ensure_numeric(self.X, "X")
        _ensure_numeric(self.x, "x")
        if not self.allow_nans:
            _ensure_no_nan(self.X, "X")
            _ensure_no_nan(self.x, "x")
        if self.y is not None:
            self.y = np.asarray(self.y)
            if self.y.shape[0] != n_samples:
                raise ValueError("y must align with X rows")
            if not self.allow_nans:
                _ensure_no_nan(self.y, "y")
        if self.metadata is None:
            self.metadata = pd.DataFrame(index=range(n_samples))
        if len(self.metadata) != n_samples:
            raise ValueError("metadata rows must match X rows")
        # Ensure consistent x grid across samples (X already aligned, but check monotonicity)
        if not np.all(np.diff(self.x) >= 0):
            raise ValueError("x grid must be sorted/non-decreasing")

    def select_by_metadata(self, key: str, value: Any) -> "SpectraSet":
        """Return subset where metadata[key] == value."""

        if key not in self.metadata.columns:
            raise KeyError(f"Metadata key '{key}' not found")
        mask = self.metadata[key] == value
        subset_X = self.X[mask.to_numpy()]
        subset_y = self.y[mask.to_numpy()] if self.y is not None else None
        subset_meta = self.metadata.loc[mask].reset_index(drop=True)
        return SpectraSet(
            X=subset_X,
            x=self.x.copy(),
            y=subset_y,
            metadata=subset_meta,
            allow_nans=self.allow_nans,
        )

    def summary_stats(self) -> Dict[str, np.ndarray]:
        """Mean and std per wavenumber."""

        return {"mean": self.X.mean(axis=0), "std": self.X.std(axis=0, ddof=0)}

    def export_to_dataframe(self) -> pd.DataFrame:
        """Flatten spectra to tabular form with metadata and optional labels."""

        df = self.metadata.copy()
        if self.y is not None:
            df["label"] = self.y
        # Use wavenumber values as column headers
        wave_cols = {f"wv_{v:g}": self.X[:, idx] for idx, v in enumerate(self.x)}
        spectra_df = pd.DataFrame(wave_cols)
        spectra_df.index = df.index
        return pd.concat([df, spectra_df], axis=1)


__all__ = ["Spectrum", "SpectraSet"]
