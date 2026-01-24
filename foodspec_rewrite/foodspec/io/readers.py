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

CSV readers for spectroscopy datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from foodspec.core.data import SpectraSet
from foodspec.core.protocol import DataSpec


def load_csv_spectra(path: str | Path, data_spec: DataSpec, allow_nans: bool = False) -> SpectraSet:
    """Load a spectroscopy dataset from CSV into a ``SpectraSet``.

    Columns expected
    ----------------
    - Optional first column for sample id (configurable via metadata_map)
    - Wavenumber columns: numeric headers (e.g., 1000, 1001.5)
    - Label column: configured via ``data_spec.label``
    - Metadata columns: configured via ``data_spec.metadata_map``

    Raises
    ------
    ValueError
        If required columns are missing or wavenumber headers are non-numeric.

    Examples
    --------
    >>> spec = DataSpec(
    ...     input="data.csv",
    ...     modality="raman",
    ...     label="target",
    ...     metadata_map={"sample_id": "id", "modality": "mod", "label": "target"},
    ... )
    >>> ds = load_csv_spectra("data.csv", spec)
    >>> ds.X.shape[0]  # number of samples
    3
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    label_col = data_spec.label
    metadata_cols = set(data_spec.metadata_map.values()) - {label_col}

    missing = [c for c in [label_col, *metadata_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Wavenumber columns are everything else
    wave_cols = [c for c in df.columns if c not in metadata_cols and c != label_col]
    if not wave_cols:
        raise ValueError("No wavenumber columns found in CSV")

    try:
        x = np.array([float(c) for c in wave_cols], dtype=float)
    except ValueError as exc:  # header not numeric
        raise ValueError("Wavenumber columns must have numeric headers") from exc

    X = df[wave_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()

    meta_df = df[list(metadata_cols)].copy() if metadata_cols else pd.DataFrame(index=df.index)
    # rename to canonical metadata keys
    rename_map = {src: canon for canon, src in data_spec.metadata_map.items() if src in meta_df.columns}
    meta_df = meta_df.rename(columns=rename_map)

    ds = SpectraSet(X=X, x=x, y=y, metadata=meta_df, allow_nans=allow_nans)
    return ds


__all__ = ["load_csv_spectra"]
