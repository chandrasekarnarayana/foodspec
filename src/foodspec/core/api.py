"""Unified FoodSpec entry point: one class to rule them all."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.core.output_bundle import OutputBundle
from foodspec.core.run_record import RunRecord, _capture_environment, _hash_data
from foodspec.io.ingest import DEFAULT_IO_REGISTRY
from foodspec.preprocess.engine import AutoPreprocess
from foodspec.features.specs import FeatureEngine, FeatureSpec
from foodspec.qc.engine import generate_qc_report


class FoodSpec:
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
        self.data, ingest_metrics, ingest_diagnostics = self._load_data(
            source, wavenumbers, metadata, modality
        )
        
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
    
    def qc(
        self,
        method: str = "robust_z",
        threshold: float = 0.5,
        **kwargs,
    ) -> FoodSpec:
        """Apply QC (quality control) to detect and flag outliers.
        
        Parameters
        ----------
        method : {'robust_z', 'mahalanobis', 'isolation_forest', 'lof'}, optional
            Outlier detection method. Default: 'robust_z'.
        threshold : float, optional
            Not used in current implementation (reserved for future scoring cutoffs).
        **kwargs
            Additional arguments forwarded to QC engine (e.g., reference_grid, batch_col, time_col).
            
        Returns
        -------
        FoodSpec
            Self (for chaining).
        """

        report = generate_qc_report(
            self.data,
            reference_grid=kwargs.get("reference_grid"),
            batch_col=kwargs.get("batch_col"),
            time_col=kwargs.get("time_col"),
            outlier_method=method,
        )

        # Record metrics and diagnostics
        self.bundle.add_metrics("qc_health", report.health.aggregates)
        self.bundle.add_metrics("qc_outliers", {"outlier_rate": float(report.outliers.labels.mean())})
        self.bundle.add_metrics("qc_drift", {"drift_score": report.drift.drift_score, "trend_slope": report.drift.trend_slope})
        self.bundle.add_diagnostic("qc_health_table", report.health.table.to_dict(orient="list"))
        self.bundle.add_diagnostic("qc_outlier_scores", report.outliers.scores.tolist())
        self.bundle.add_diagnostic("qc_recommendation", report.recommendations)

        step_hash = hashlib.sha256(json.dumps({"method": method, "recommendation": report.recommendations}, sort_keys=True).encode()).hexdigest()[:8]
        self.bundle.run_record.add_step(
            "qc",
            step_hash,
            metadata={"method": method, "recommendation": report.recommendations},
        )

        self._steps_applied.append("qc")
        return self
    
    def preprocess(
        self,
        preset: str = "auto",
        **kwargs,
    ) -> FoodSpec:
        """Apply preprocessing pipeline.
        
        Parameters
        ----------
        preset : str, optional
            Preset name: 'auto', 'quick', 'standard', 'publication'. Default: 'auto'.
        **kwargs
            Override preset parameters (forwarded to AutoPreprocess or manual presets).
            
        Returns
        -------
        FoodSpec
            Self (for chaining).
        """

        config_dict = {"preset": preset, **kwargs}

        # Use AutoPreprocess as the default preset (Phase 3).
        if preset == "auto":
            auto = AutoPreprocess(
                baselines=kwargs.get("baselines"),
                smoothers=kwargs.get("smoothers"),
                normalizers=kwargs.get("normalizers"),
                derivatives=kwargs.get("derivatives"),
            )
            result = auto.search(self.data)
            processed, metrics = result.pipeline.transform(self.data)
            self.data = processed

            # Record metrics and explanation
            self.bundle.add_metrics("preprocess", metrics)
            self.bundle.add_diagnostic("preprocess", {"explanation": result.explanation})

            step_hash = result.pipeline.hash()
            self.bundle.run_record.add_step(
                "preprocess",
                step_hash,
                metadata={"preset": preset, "pipeline": result.pipeline.to_dict(), "metrics": metrics},
            )
        else:
            # Simple fallback presets (quick/standard/publication) using AutoPreprocess with narrow grids
            preset_map = {
                "quick": {
                    "baselines": [{"method": "rubberband"}],
                    "smoothers": [{"method": "moving_average", "window": 3}],
                    "normalizers": [{"method": "snv"}],
                    "derivatives": [{"order": 0}],
                },
                "standard": {
                    "baselines": [{"method": "als", "lam": 1e5, "p": 0.01}],
                    "smoothers": [{"method": "savgol", "window_length": 7, "polyorder": 3}],
                    "normalizers": [{"method": "vector"}],
                    "derivatives": [{"order": 1, "window_length": 9, "polyorder": 2}],
                },
                "publication": {
                    "baselines": [{"method": "als", "lam": 1e6, "p": 0.001}],
                    "smoothers": [{"method": "savgol", "window_length": 11, "polyorder": 3}],
                    "normalizers": [{"method": "msc"}],
                    "derivatives": [{"order": 1, "window_length": 15, "polyorder": 3}],
                },
            }
            preset_cfg = preset_map.get(preset)
            if preset_cfg is None:
                raise ValueError(f"Unknown preprocessing preset: {preset}")
            auto = AutoPreprocess(**preset_cfg)
            result = auto.search(self.data)
            processed, metrics = result.pipeline.transform(self.data)
            self.data = processed
            self.bundle.add_metrics("preprocess", metrics)
            self.bundle.add_diagnostic("preprocess", {"explanation": result.explanation})
            step_hash = result.pipeline.hash()
            self.bundle.run_record.add_step(
                "preprocess",
                step_hash,
                metadata={"preset": preset, "pipeline": result.pipeline.to_dict(), "metrics": metrics},
            )

        self._steps_applied.append(f"preprocess({preset})")
        return self
    
    def features(
        self,
        preset: str = "specs",
        specs: Optional[list[FeatureSpec]] = None,
        **kwargs,
    ) -> FoodSpec:
        """Extract features (peaks, ratios, etc.).
        
        Parameters
        ----------
        preset : str, optional
            Preset name. Default: 'specs' uses provided FeatureSpec list.
        specs : list[FeatureSpec], optional
            Feature specifications to evaluate when preset='specs'.
        **kwargs
            Override preset parameters or provide default specs for other presets.
            
        Returns
        -------
        FoodSpec
            Self (for chaining).
        """
        if preset == "specs":
            if not specs:
                raise ValueError("Feature specs are required when preset='specs'")
            engine = FeatureEngine(specs)
            features_df, diag = engine.evaluate(self.data)
        else:
            # Minimal defaults for quick/standard presets using bands and peak heights
            default_specs = {
                "quick": [
                    FeatureSpec(name="band_full", ftype="band", regions=[(float(self.data.wavenumbers.min()), float(self.data.wavenumbers.max()))]),
                ],
                "standard": [
                    FeatureSpec(name="band_mid", ftype="band", regions=[(float(np.percentile(self.data.wavenumbers, 25)), float(np.percentile(self.data.wavenumbers, 75)))]),
                    FeatureSpec(name="peak_max", ftype="peak", regions=[(float(self.data.wavenumbers[np.argmax(self.data.x.mean(axis=0))]) - 5, float(self.data.wavenumbers[np.argmax(self.data.x.mean(axis=0))]) + 5)], params={"tolerance": 5.0, "metrics": ("height",)}),
                ],
            }
            chosen = default_specs.get(preset)
            if chosen is None:
                raise ValueError(f"Unknown features preset: {preset}")
            engine = FeatureEngine(chosen)
            features_df, diag = engine.evaluate(self.data)

        self.bundle.add_diagnostic("features_table", features_df.to_dict(orient="list"))
        self.bundle.add_metrics("features", {"n_features": features_df.shape[1]})

        spec_hashes = [s.hash() for s in (specs or [])]
        step_hash = hashlib.sha256(json.dumps({"preset": preset, "spec_hashes": spec_hashes}, sort_keys=True).encode()).hexdigest()[:8]
        self.bundle.run_record.add_step(
            "features",
            step_hash,
            metadata={"preset": preset, "features": spec_hashes or ["default"]},
        )

        self._steps_applied.append(f"features({preset})")
        return self
    
    def train(
        self,
        algorithm: str = "rf",
        label_column: str = "label",
        test_size: float = 0.3,
        cv_folds: int = 5,
        **kwargs,
    ) -> FoodSpec:
        """Train a model on the data.
        
        Parameters
        ----------
        algorithm : {'rf', 'lr', 'svm', 'pls_da'}, optional
            Algorithm to use. Default: 'rf'.
        label_column : str, optional
            Metadata column for labels. Default: 'label'.
        test_size : float, optional
            Test set fraction. Default: 0.3.
        cv_folds : int, optional
            Cross-validation folds. Default: 5.
        **kwargs
            Algorithm-specific parameters.
            
        Returns
        -------
        FoodSpec
            Self (for chaining).
        """
        from foodspec.chemometrics.models import make_classifier
        from foodspec.chemometrics.validation import compute_classification_metrics
        
        # Extract features and labels
        X, y = self.data.to_X_y(target_col=label_column)
        
        # Train model
        pipeline = make_classifier(
            X,
            y,
            algorithm=algorithm,
            cv_folds=cv_folds,
            **kwargs,
        )
        
        # Compute metrics
        cv_metrics = compute_classification_metrics(pipeline, X, y, cv=cv_folds)
        
        # Store model and metrics
        self.bundle.add_artifact("model", pipeline)
        self.bundle.add_metrics("cv_metrics", cv_metrics)
        
        # Log step
        self.bundle.run_record.add_step(
            "train",
            hashlib.sha256(json.dumps({"algorithm": algorithm}).encode()).hexdigest()[:8],
            metadata={"algorithm": algorithm, "cv_folds": cv_folds},
        )
        
        self._steps_applied.append(f"train({algorithm})")
        return self
    
    def export(
        self,
        path: Optional[Union[str, Path]] = None,
        formats: Optional[list] = None,
    ) -> Path:
        """Export all outputs to disk.
        
        Parameters
        ----------
        path : Path or str, optional
            Output directory. If None, uses self.output_dir.
        formats : list, optional
            Export formats. Default: ['json', 'csv', 'png', 'joblib'].
            
        Returns
        -------
        Path
            Directory containing all outputs.
        """
        path = path or self.output_dir
        return self.bundle.export(path, formats=formats)
    
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
