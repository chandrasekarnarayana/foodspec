"""I/O mixin for FoodSpec API - data loading functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from foodspec.data_objects.spectra_set import FoodSpectrumSet
from foodspec.io.ingest import DEFAULT_IO_REGISTRY


class FoodSpecIOMixin:
    """Mixin class providing data loading capabilities for FoodSpec."""

    @staticmethod
    def _load_data(
        source: Union[str, Path, FoodSpectrumSet, np.ndarray, pd.DataFrame],
        wavenumbers: Optional[np.ndarray],
        metadata: Optional[pd.DataFrame],
        modality: str,
    ) -> tuple[FoodSpectrumSet, Dict[str, Any], Dict[str, Any]]:
        """Load data from various sources into FoodSpectrumSet + ingestion metrics."""

        # Already a FoodSpectrumSet
        if isinstance(source, FoodSpectrumSet):
            return source, {}, {}

        # NumPy array
        if isinstance(source, np.ndarray):
            if wavenumbers is None:
                raise ValueError("wavenumbers required for np.ndarray source")
            if metadata is None:
                metadata = pd.DataFrame({"sample_id": [f"s{i}" for i in range(source.shape[0])]})
            ds = FoodSpectrumSet(x=source, wavenumbers=wavenumbers, metadata=metadata, modality=modality)
            return ds, {}, {}

        # Pandas DataFrame (wide format)
        if isinstance(source, pd.DataFrame):
            if wavenumbers is None:
                wn_col = source.iloc[:, 0]
                spec_data = source.iloc[:, 1:].to_numpy()
                wavenumbers = wn_col.to_numpy()
                metadata = pd.DataFrame({"sample_id": source.iloc[:, 1:].columns.astype(str)})
            else:
                spec_data = source.to_numpy()
            ds = FoodSpectrumSet(x=spec_data, wavenumbers=wavenumbers, metadata=metadata, modality=modality)
            return ds, {}, {}

        # Path-based sources (file or folder): use ingestion registry
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        ingest_result = DEFAULT_IO_REGISTRY.load("auto", source_path, modality=modality)
        return ingest_result.dataset, ingest_result.metrics, ingest_result.diagnostics

    @staticmethod
    def _dataframe_to_spectrum_set(df: pd.DataFrame, modality: str) -> FoodSpectrumSet:
        """Convert wide DataFrame back to FoodSpectrumSet.

        Assumes first column is wavenumbers (or metadata columns before first numeric).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns (wavenumbers + spectrum)")

        wn = df[numeric_cols[0]].to_numpy()
        spectra = df[numeric_cols[1:]].to_numpy()
        metadata = pd.DataFrame({"sample_id": numeric_cols[1:]})

        return FoodSpectrumSet(x=spectra, wavenumbers=wn, metadata=metadata, modality=modality)
