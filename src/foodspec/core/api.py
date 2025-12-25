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

    def library_similarity(
        self,
        library: FoodSpectrumSet,
        metric: Literal["euclidean", "cosine", "pearson", "sid", "sam"] = "cosine",
        top_k: int = 5,
        add_conf: bool = True,
    ) -> pd.DataFrame:
        """Run a similarity search against a reference library and record outputs.

        Records a similarity table and an overlay plot (query 0 vs. its top match)
        into the OutputBundle diagnostics.

        Parameters
        ----------
        library : FoodSpectrumSet
            Reference spectra library.
        metric : str
            Distance metric (euclidean, cosine, pearson, sid, sam). Default 'cosine'.
        top_k : int
            Number of top matches to report per query. Default 5.
        add_conf : bool
            If True, append confidence and decision columns to the similarity table.

        Returns
        -------
        pandas.DataFrame
            Similarity table with distances (and confidence/decision if enabled).
        """
        from foodspec.features.library import LibraryIndex, overlay_plot
        from foodspec.features.confidence import add_confidence

        # Build library index and compute similarity table
        lib = LibraryIndex.from_dataset(library)
        query_ids = list(
            self.data.metadata.get("sample_id", pd.Series(np.arange(len(self.data))).astype(str))
        )
        sim_table = lib.search(self.data.x, metric=metric, top_k=top_k, query_ids=query_ids)

        # Confidence and decision mapping
        if add_conf:
            sim_table = add_confidence(sim_table, metric=metric)

        # Add diagnostics
        self.bundle.add_diagnostic("similarity_table", sim_table)

        # Overlay plot for first query vs its top-1 match (if any)
        try:
            first_q = 0
            top_row = sim_table[sim_table["query_index"] == first_q].sort_values("rank").iloc[0]
            fig, ax = overlay_plot(
                self.data.x[first_q],
                lib.X[int(top_row["library_index"])],
                self.data.wavenumbers,
            )
            self.bundle.add_diagnostic("overlay_query0_top1", fig)
        except Exception:
            # Non-critical; skip if plotting fails
            pass

        # Log step
        self.bundle.run_record.add_step(
            "library_similarity",
            hashlib.sha256(json.dumps({"metric": metric, "top_k": top_k}, sort_keys=True).encode()).hexdigest()[:8],
            metadata={"metric": metric, "top_k": top_k},
        )
        self._steps_applied.append(f"library_similarity({metric},k={top_k})")

        return sim_table
    
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
    
    # ──────────────────────────────────────────────────────────────────────────
    # Moat Methods: Matrix Correction, Heating Trajectory, Calibration Transfer
    # ──────────────────────────────────────────────────────────────────────────
    
    def apply_matrix_correction(
        self,
        method: Literal["background_air", "background_dark", "adaptive_baseline", "none"] = "adaptive_baseline",
        scaling: Literal["median_mad", "huber", "mcd", "none"] = "median_mad",
        domain_adapt: bool = False,
        matrix_column: Optional[str] = None,
        reference_spectra: Optional[np.ndarray] = None,
    ) -> FoodSpec:
        """
        Apply matrix correction to remove matrix effects (e.g., chips vs. pure oil).
        
        **Key Assumptions:**
        - Background reference spectra measured under identical conditions
        - Matrix types known/inferrable from metadata
        - Domain adaptation requires ≥2 matrix types with ≥10 samples each
        - Spectral ranges aligned before correction
        
        See foodspec.matrix_correction module docstring for full details.
        
        Parameters
        ----------
        method : str, default='adaptive_baseline'
            Background subtraction method.
        scaling : str, default='median_mad'
            Robust scaling method per matrix type.
        domain_adapt : bool, default=False
            Whether to apply subspace alignment between matrices.
        matrix_column : str, optional
            Metadata column with matrix type labels.
        reference_spectra : np.ndarray, optional
            Background reference (for background_air/dark methods).
            
        Returns
        -------
        FoodSpec
            Self (for chaining).
        """
        from foodspec.preprocess.matrix_correction import apply_matrix_correction as _apply_mc
        
        self.data, mc_metrics = _apply_mc(
            self.data,
            method=method,
            scaling=scaling,
            domain_adapt=domain_adapt,
            matrix_column=matrix_column,
            reference_spectra=reference_spectra,
        )
        
        # Record metrics
        for key, val in mc_metrics.items():
            self.bundle.add_metrics(f"matrix_correction_{key}", val)
        
        self.bundle.run_record.add_step(
            "matrix_correction",
            hashlib.sha256(json.dumps(mc_metrics, sort_keys=True).encode()).hexdigest()[:8],
            metadata={"method": method, "scaling": scaling, "domain_adapt": domain_adapt},
        )
        self._steps_applied.append("matrix_correction")
        
        return self
    
    def analyze_heating_trajectory(
        self,
        time_column: str,
        indices: List[str] = ["pi", "tfc", "oit_proxy"],
        classify_stages: bool = False,
        stage_column: Optional[str] = None,
        estimate_shelf_life: bool = False,
        shelf_life_threshold: Optional[float] = None,
        shelf_life_index: str = "pi",
    ) -> Dict[str, Any]:
        """
        Analyze heating/oxidation trajectory from time-series spectra.
        
        **Key Assumptions:**
        - time_column exists and is numeric (hours, days, timestamps)
        - Repeated measurements over time (longitudinal data)
        - Degradation is monotonic or follows known patterns
        - ≥5 time points per sample/group for reliable regression
        - No major batch effects confounding time trends
        
        See foodspec.workflows.heating_trajectory module docstring for full details.
        
        Parameters
        ----------
        time_column : str
            Metadata column with time values.
        indices : list of str, default=['pi', 'tfc', 'oit_proxy']
            Oxidation indices to extract and model.
        classify_stages : bool, default=False
            Whether to train degradation stage classifier.
        stage_column : str, optional
            Metadata column with stage labels (required if classify_stages=True).
        estimate_shelf_life : bool, default=False
            Whether to estimate shelf life.
        shelf_life_threshold : float, optional
            Threshold for shelf-life criterion (required if estimate_shelf_life=True).
        shelf_life_index : str, default='pi'
            Index to use for shelf-life estimation.
            
        Returns
        -------
        results : dict
            - 'indices': extracted indices DataFrame
            - 'trajectory_models': fit metrics per index
            - 'stage_classification' (if enabled): classification metrics
            - 'shelf_life' (if enabled): shelf-life estimation
        """
        from foodspec.workflows.heating_trajectory import analyze_heating_trajectory as _analyze_ht
        
        results = _analyze_ht(
            self.data,
            time_column=time_column,
            indices=indices,
            classify_stages=classify_stages,
            stage_column=stage_column,
            estimate_shelf_life=estimate_shelf_life,
            shelf_life_threshold=shelf_life_threshold,
            shelf_life_index=shelf_life_index,
        )

        # Provide backwards-compatible key expected by tests
        if "trajectory" not in results:
            results["trajectory"] = results.get("trajectory_models", {})
        
        # Record metrics
        self.bundle.add_metrics("heating_trajectory", results.get("trajectory_models", {}))
        if "stage_classification" in results:
            self.bundle.add_metrics("stage_classification", results["stage_classification"]["metrics"])
        if "shelf_life" in results:
            self.bundle.add_metrics("shelf_life", results["shelf_life"])
        
        self.bundle.run_record.add_step(
            "heating_trajectory",
            hashlib.sha256(json.dumps(results.get("trajectory_models", {}), sort_keys=True).encode()).hexdigest()[:8],
            metadata={"time_column": time_column, "indices": indices},
        )
        self._steps_applied.append("heating_trajectory")
        
        return results
    
    def apply_calibration_transfer(
        self,
        source_standards: np.ndarray,
        target_standards: np.ndarray,
        method: Literal["ds", "pds"] = "ds",
        pds_window_size: int = 11,
        alpha: float = 1.0,
    ) -> FoodSpec:
        """
        Apply calibration transfer to align target instrument to source.
        
        **Key Assumptions:**
        - Source/target standards are paired (same samples measured on both)
        - Standards span the calibration range
        - Spectral alignment already performed
        - Linear transformation adequate for instrument differences
        
        See foodspec.calibration_transfer module docstring for full details.
        
        Parameters
        ----------
        source_standards : np.ndarray, shape (n_standards, n_wavenumbers)
            Source (reference) instrument spectra.
        target_standards : np.ndarray, shape (n_standards, n_wavenumbers)
            Target (slave) instrument spectra.
        method : {'ds', 'pds'}, default='ds'
            Transfer method (Direct Standardization or Piecewise DS).
        pds_window_size : int, default=11
            PDS window size (ignored if method='ds').
        alpha : float, default=1.0
            Ridge regularization parameter.
            
        Returns
        -------
        FoodSpec
            Self (for chaining).
        """
        from foodspec.preprocess.calibration_transfer import calibration_transfer_workflow
        
        self.data.x, ct_metrics = calibration_transfer_workflow(
            source_standards,
            target_standards,
            self.data.x,
            method=method,
            pds_window_size=pds_window_size,
            alpha=alpha,
        )
        
        # Record metrics
        for key, val in ct_metrics.items():
            self.bundle.add_metrics(f"calibration_transfer_{key}", val)
        # Generate simple HTML dashboard if metrics include success metrics
        try:
            from foodspec.calibration_transfer_dashboard import build_dashboard_html
            html = build_dashboard_html(ct_metrics)
            self.bundle.add_diagnostic("calibration_transfer_dashboard", html)
        except Exception:
            pass
        
        self.bundle.run_record.add_step(
            "calibration_transfer",
            hashlib.sha256(json.dumps(ct_metrics, sort_keys=True).encode()).hexdigest()[:8],
            metadata={"method": method, "pds_window_size": pds_window_size},
        )
        self._steps_applied.append("calibration_transfer")
        
        return self
    
    # -------------------------------------------------------------------------
    # Data Governance & Dataset Intelligence (Moat 4)
    # -------------------------------------------------------------------------
    
    def summarize_dataset(
        self,
        label_column: Optional[str] = None,
        required_metadata_columns: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive dataset summary for at-a-glance quality assessment.
        
        **Returns:**
        - class_distribution (if label_column provided)
        - spectral_quality (SNR, range, NaN/inf counts)
        - metadata_completeness
        - dataset_info (n_samples, n_wavenumbers, modality)
        
        See foodspec.core.summary module docstring for details.
        
        Parameters
        ----------
        label_column : str, optional
            Column with class labels.
        required_metadata_columns : list of str, optional
            Metadata columns that must be present.
            
        Returns
        -------
        summary : dict
            Comprehensive dataset summary.
        """
        from foodspec.core.summary import summarize_dataset
        
        summary = summarize_dataset(
            self.data,
            label_column=label_column,
            required_metadata_columns=required_metadata_columns,
        )
        
        # Record summary metrics
        self.bundle.add_metrics("dataset_summary", summary)
        
        return summary
    
    def check_class_balance(
        self,
        label_column: str,
        severe_threshold: float = 10.0,
        min_samples_per_class: int = 20,
    ) -> Dict[str, Any]:
        """
        Check class balance and flag severe imbalance.
        
        **Returns:**
        - samples_per_class, imbalance_ratio, severe_imbalance flag
        - undersized_classes, recommended_action
        
        See foodspec.qc.dataset_qc module docstring for details.
        
        Parameters
        ----------
        label_column : str
            Column with class labels.
        severe_threshold : float, default=10.0
            Imbalance ratio above which to flag as severe.
        min_samples_per_class : int, default=20
            Minimum recommended samples per class.
            
        Returns
        -------
        metrics : dict
            Class balance diagnostics.
        """
        from foodspec.qc.dataset_qc import check_class_balance
        
        balance = check_class_balance(
            self.data.metadata,
            label_column,
            severe_threshold=severe_threshold,
            min_samples_per_class=min_samples_per_class,
        )
        
        self.bundle.add_metrics("class_balance", balance)
        
        return balance
    
    def assess_replicate_consistency(
        self,
        replicate_column: str,
        technical_cv_threshold: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Compute coefficient of variation (CV) for replicate groups.
        
        **Returns:**
        - cv_per_replicate, median_cv, high_variability_replicates
        
        See foodspec.qc.replicates module docstring for details.
        
        Parameters
        ----------
        replicate_column : str
            Column defining replicate groups.
        technical_cv_threshold : float, default=10.0
            CV (%) above which to flag as high variability.
            
        Returns
        -------
        metrics : dict
            Replicate consistency metrics.
        """
        from foodspec.qc.replicates import compute_replicate_consistency
        
        consistency = compute_replicate_consistency(
            self.data.x,
            self.data.metadata,
            replicate_column,
            technical_cv_threshold=technical_cv_threshold,
        )
        
        self.bundle.add_metrics("replicate_consistency", consistency)
        
        return consistency
    
    def detect_leakage(
        self,
        label_column: str,
        batch_column: Optional[str] = None,
        replicate_column: Optional[str] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Detect data leakage: batch–label correlation and replicate splits.
        
        **Returns:**
        - batch_label_correlation (Cramér's V)
        - replicate_leakage (risk/detection)
        - overall_risk: 'high', 'moderate', 'low'
        
        See foodspec.qc.leakage module docstring for details.
        
        Parameters
        ----------
        label_column : str
            Column with class labels.
        batch_column : str, optional
            Column defining batches.
        replicate_column : str, optional
            Column defining replicate groups.
        train_indices : np.ndarray, optional
            Row indices for training set.
        test_indices : np.ndarray, optional
            Row indices for test set.
            
        Returns
        -------
        leakage_report : dict
            Comprehensive leakage diagnostics.
        """
        from foodspec.qc.leakage import detect_leakage
        
        leakage_report = detect_leakage(
            self.data,
            label_column,
            batch_column=batch_column,
            replicate_column=replicate_column,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        
        self.bundle.add_metrics("leakage_detection", leakage_report)
        
        return leakage_report
    
    def compute_readiness_score(
        self,
        label_column: str,
        batch_column: Optional[str] = None,
        replicate_column: Optional[str] = None,
        required_metadata_columns: Optional[list] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive dataset readiness score (0-100).
        
        **Scoring Dimensions:**
        - Sample size, class balance, replicate consistency
        - Metadata completeness, spectral quality, leakage risk
        
        **Returns:**
        - overall_score: 0-100
        - dimension_scores: individual dimension scores
        - passed_criteria, failed_criteria
        - recommendation: text guidance
        
        See foodspec.qc.readiness module docstring for details.
        
        Parameters
        ----------
        label_column : str
            Column with class labels.
        batch_column : str, optional
            Column defining batches.
        replicate_column : str, optional
            Column defining replicate groups.
        required_metadata_columns : list of str, optional
            Metadata columns that must be complete.
        weights : dict, optional
            Custom weights for scoring dimensions.
            
        Returns
        -------
        score_report : dict
            Readiness score report.
        """
        from foodspec.qc.readiness import compute_readiness_score
        
        score_report = compute_readiness_score(
            self.data,
            label_column,
            batch_column=batch_column,
            replicate_column=replicate_column,
            required_metadata_columns=required_metadata_columns,
            weights=weights,
        )
        
        self.bundle.add_metrics("readiness_score", score_report)
        
        return score_report
