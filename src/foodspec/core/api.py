"""Unified FoodSpec entry point: one class to rule them all."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from foodspec.core.api_diagnostics import FoodSpecDiagnosticsMixin

# Import all mixins
from foodspec.core.api_io import FoodSpecIOMixin
from foodspec.core.api_modeling import FoodSpecModelingMixin
from foodspec.core.api_preprocess import FoodSpecPreprocessMixin
from foodspec.core.api_workflows import FoodSpecWorkflowsMixin
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.core.output_bundle import OutputBundle
from foodspec.core.run_record import RunRecord, _capture_environment, _hash_data


class FoodSpec(
    FoodSpecIOMixin,
    FoodSpecPreprocessMixin,
    FoodSpecModelingMixin,
    FoodSpecWorkflowsMixin,
    FoodSpecDiagnosticsMixin,
):
    """Unified entry point for FoodSpec workflows: load → preprocess → feature → train → export.

    Single class provides chainable UX for the entire spectroscopy pipeline.

    Parameters
    ----------
    source : str, Path, FoodSpectrumSet, np.ndarray, or pd.DataFrame
        Data source:
        - Path/str: file or folder path (auto-detected format)
        - FoodSpectrumSet: existing dataset
        - np.ndarray: spectral intensity array (n_samples, n_wavenumbers)
        - pd.DataFrame: wide format (first col=wavenumbers, rest=spectra)
    wavenumbers : np.ndarray, optional
        X-axis (wavenumbers). Required if source is np.ndarray.
    metadata : pd.DataFrame, optional
        Sample metadata. Required if source is np.ndarray.
    modality : {'raman', 'ftir', 'nir'}, optional
        Spectroscopy modality. Default: 'raman'.
    kind : str, optional
        Descriptive name for this dataset.
    output_dir : Path or str, optional
        Directory for outputs. If None, defaults to ./foodspec_runs/.

    Attributes
    ----------
    data : FoodSpectrumSet
        Current spectral dataset.
    output_dir : Path
        Output directory.
    bundle : OutputBundle
        Artifact bundle (metrics, diagnostics, provenance).
    config : dict
        Configuration for reproducibility.

    Examples
    --------
    >>> fs = FoodSpec("oils.csv", modality="raman")
    >>> fs.qc().preprocess("standard").features("oil_auth").train("rf", label_column="oil_type")
    >>> artifacts = fs.bundle.export(output_dir="./results/")
    """

    def __init__(
        self,
        source: Union[str, Path, FoodSpectrumSet, np.ndarray, pd.DataFrame],
        wavenumbers: Optional[np.ndarray] = None,
        metadata: Optional[pd.DataFrame] = None,
        modality: Literal["raman", "ftir", "nir"] = "raman",
        kind: str = "spectral_data",
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize FoodSpec with data source."""

        # Load data via ingestion registry (captures I/O quality metrics)
        self.data, ingest_metrics, ingest_diagnostics = self._load_data(source, wavenumbers, metadata, modality)

        # Setup output directory
        self.output_dir = Path(output_dir or "foodspec_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize configuration
        self.config = {
            "kind": kind,
            "modality": modality,
            "source": str(source) if isinstance(source, (str, Path)) else type(source).__name__,
        }

        # Compute dataset hash
        dataset_hash = _hash_data(self.data.x)

        # Create output bundle with run record
        run_record = RunRecord(
            workflow_name="foodspec",
            config=self.config,
            dataset_hash=dataset_hash,
            environment=_capture_environment(),
        )
        self.bundle = OutputBundle(run_record=run_record)

        if ingest_metrics:
            self.bundle.add_metrics("ingest", ingest_metrics)
        if ingest_diagnostics:
            self.bundle.add_diagnostic("ingest", ingest_diagnostics)
        if ingest_metrics:
            self.bundle.run_record.add_step(
                "ingest",
                hashlib.sha256(json.dumps(ingest_metrics, sort_keys=True).encode()).hexdigest()[:8],
                metadata=ingest_metrics,
            )

        # Pipeline tracking
        self._steps_applied = []

    def summary(self) -> str:
        """Generate summary of the workflow.

        Returns
        -------
        str
            Human-readable summary.
        """
        lines = [
            "FoodSpec Workflow Summary",
            "=" * 50,
            f"Dataset: {self.data.modality}, n={len(self.data)}, n_features={self.data.x.shape[1]}",
            f"Steps applied: {', '.join(self._steps_applied) if self._steps_applied else 'None'}",
            "",
            self.bundle.summary(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FoodSpec(modality={self.data.modality}, n={len(self.data)}, "
            f"steps={len(self._steps_applied)}, output_dir={self.output_dir})"
        )
