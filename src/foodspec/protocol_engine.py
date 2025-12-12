"""
Vendor-neutral Auto-Protocol Engine for FoodSpec
------------------------------------------------

Minimal but extensible engine that loads a protocol description (YAML/JSON),
executes preprocessing + analysis steps, and bundles outputs (tables, figures,
reports, metadata, logs).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from sklearn.ensemble import RandomForestClassifier

from foodspec.output_bundle import (
    append_log,
    create_run_folder,
    save_figures,
    save_index,
    save_metadata,
    save_report_html,
    save_report_text,
    save_tables,
)
from foodspec.preprocessing_pipeline import PreprocessingConfig, detect_input_mode, run_full_preprocessing
from foodspec.rq import PeakDefinition, RatioDefinition, RatioQualityEngine, RQConfig
from foodspec.spectral_dataset import HyperspectralDataset, SpectralDataset


@dataclass
class ProtocolConfig:
    name: str
    description: str = ""
    when_to_use: str = ""
    version: str = "0.1.0"
    min_foodspec_version: Optional[str] = None
    seed: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_columns: Dict[str, str] = field(default_factory=dict)
    report_templates: Dict[str, str] = field(default_factory=dict)
    required_metadata: List[str] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    validation_strategy: str = "standard"  # standard | batch_aware | group_stratified

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProtocolConfig":
        return ProtocolConfig(
            name=d.get("name", "Unnamed_Protocol"),
            description=d.get("description", ""),
            when_to_use=d.get("when_to_use", ""),
            version=d.get("version", d.get("protocol_version", "0.1.0")),
            min_foodspec_version=d.get("min_foodspec_version"),
            seed=d.get("seed", 0),
            steps=d.get("steps", []),
            expected_columns=d.get("expected_columns", {}),
            report_templates=d.get("report_templates", {}),
            required_metadata=d.get("required_metadata", []),
            inputs=d.get("inputs", []),
            validation_strategy=d.get("validation_strategy", "standard"),
        )

    @staticmethod
    def from_file(path: Union[str, Path]) -> "ProtocolConfig":
        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise ImportError("PyYAML not installed.")
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        return ProtocolConfig.from_dict(payload)


@dataclass
class ProtocolRunResult:
    run_dir: Optional[Path]
    logs: List[str]
    metadata: Dict[str, Any]
    tables: Dict[str, pd.DataFrame]
    figures: Dict[str, Any]
    report: str
    summary: str


class Step:
    name: str = "base_step"

    def run(self, ctx: Dict[str, Any]):
        raise NotImplementedError


class PreprocessStep(Step):
    name = "preprocess"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        df: pd.DataFrame = ctx["data"]
        if ctx.get("cancel"):
            ctx["logs"].append("[preprocess] cancelled")
            return
        peaks_cfg = self.cfg.get("peaks", [])
        peak_defs = [
            PeakDefinition(
                name=p.get("name"),
                column=p.get("column", p.get("name")),
                wavenumber=p.get("wavenumber"),
                window=tuple(p.get("window")) if p.get("window") else None,
            )
            for p in peaks_cfg
        ]
        pp_cfg = PreprocessingConfig(
            baseline_lambda=self.cfg.get("baseline_lambda", 1e5),
            baseline_p=self.cfg.get("baseline_p", 0.01),
            baseline_enabled=self.cfg.get("baseline_enabled", True),
            smooth_enabled=self.cfg.get("smooth_enabled", True),
            smooth_window=self.cfg.get("smooth_window", 7),
            smooth_polyorder=self.cfg.get("smooth_polyorder", 3),
            normalization=self.cfg.get("normalization", "reference"),
            reference_wavenumber=self.cfg.get("reference_wavenumber", 2720.0),
            peak_definitions=peak_defs,
        )
        ctx["logs"].append(f"[preprocess] mode={detect_input_mode(df)}, cfg={pp_cfg}")
        ctx["data"] = run_full_preprocessing(df, pp_cfg)
        ctx["metadata"]["preprocessing"] = pp_cfg.__dict__


class RQAnalysisStep(Step):
    name = "rq_analysis"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        peaks_cfg = self.cfg.get("peaks", [])
        ratios_cfg = self.cfg.get("ratios", [])
        peaks = [
            PeakDefinition(
                name=p["name"],
                column=p.get("column", p["name"]),
                wavenumber=p.get("wavenumber"),
                window=tuple(p.get("window")) if p.get("window") else None,
            )
            for p in peaks_cfg
        ]
        ratios = [
            RatioDefinition(
                name=r["name"],
                numerator=r["numerator"],
                denominator=r["denominator"],
            )
            for r in ratios_cfg
        ]
        rq_cfg = RQConfig(
            oil_col=self.cfg.get("oil_col", "oil_type"),
            matrix_col=self.cfg.get("matrix_col", "matrix"),
            heating_col=self.cfg.get("heating_col", "heating_stage"),
            random_state=self.cfg.get("random_state", 0),
            n_splits=self.cfg.get("n_splits", 5),
            normalization_modes=self.cfg.get("normalization_modes", ["reference"]),
            minimal_panel_target_accuracy=self.cfg.get("minimal_panel_target_accuracy", 0.9),
            enable_clustering=self.cfg.get("enable_clustering", True),
            adjust_p_values=self.cfg.get("adjust_p_values", True),
        )
        engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=rq_cfg)
        # Optional validation metrics
        validation_metrics = None
        try:
            from foodspec.validation import nested_cv

            df = ctx["data"]
            feature_cols = [p.column for p in peaks] + [r.name for r in ratios]
            feature_cols = [c for c in feature_cols if c in df.columns]
            label_col = rq_cfg.oil_col
            if feature_cols and label_col in df.columns:
                X = df[feature_cols].astype(float).to_numpy()
                y = df[label_col].astype(str).to_numpy()
                class_counts = pd.Series(y).value_counts()
                if (class_counts < 2).any():
                    validation_metrics = None
                    raise ValueError("Too few samples per class for CV; skipping validation metrics.")
                groups = None
                if self.cfg.get("validation_strategy") == "batch_aware" and self.cfg.get("batch_col") in df.columns:
                    groups = df[self.cfg.get("batch_col")].to_numpy()
                results = nested_cv(
                    RandomForestClassifier(n_estimators=150, random_state=rq_cfg.random_state),
                    X,
                    y,
                    groups=groups,
                    outer_splits=max(2, min(5, len(np.unique(y)))),
                    inner_splits=3,
                )
                if results:
                    # Aggregate metrics
                    bal_acc = float(np.mean([r["bal_accuracy"] for r in results]))
                    recalls = np.mean([r["per_class_recall"] for r in results], axis=0).tolist()
                    aucs = [r.get("roc_auc") for r in results if r.get("roc_auc") is not None]
                    validation_metrics = {
                        "balanced_accuracy": bal_acc,
                        "per_class_recall": recalls,
                        "roc_auc": float(np.mean(aucs)) if aucs else None,
                    }
        except Exception:
            validation_metrics = None

        res = engine.run_all(ctx["data"], validation_metrics=validation_metrics)
        ctx["tables"].update(
            {
                "stability_summary": res.stability_summary,
                "discriminative_summary": res.discriminative_summary,
                "feature_importance": res.feature_importance,
                "heating_trend_summary": res.heating_trend_summary,
                "oil_vs_chips_summary": res.oil_vs_chips_summary,
                "normalization_comparison": res.normalization_comparison,
                "minimal_panel": res.minimal_panel,
            }
        )
        ctx["figures"].update(res.figures if hasattr(res, "figures") else {})
        ctx["report"] = res.text_report
        ctx["summary"] = " ".join(res.text_report.splitlines()[:10])
        ctx["logs"].append("[rq_analysis] completed")


class OutputStep(Step):
    name = "output"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        out_dir = Path(self.cfg.get("output_dir", "protocol_runs"))
        run_dir = create_run_folder(out_dir)
        ctx["run_dir"] = run_dir
        save_report_text(run_dir / "report.txt", ctx.get("report", ""))
        save_report_html(run_dir / "report.html", ctx.get("report", ""))
        save_tables(run_dir, ctx.get("tables", {}))
        save_figures(run_dir, ctx.get("figures", {}))
        ctx["metadata"]["logs"] = ctx["logs"]
        save_metadata(run_dir, ctx["metadata"])
        save_index(
            run_dir,
            ctx["metadata"],
            ctx.get("tables", {}),
            ctx.get("figures", {}),
            ctx.get("validation", {}).get("warnings", []),
        )
        # Persist HSI artifacts when available
        hsi_labels = ctx.get("hsi_labels")
        if hsi_labels is not None:
            np.save(run_dir / "hsi" / "label_map.npy", hsi_labels)
        hsi_obj = ctx.get("hsi")
        if hsi_obj is not None and getattr(hsi_obj, "roi_masks", None):
            for name, mask in hsi_obj.roi_masks.items():
                np.save(run_dir / "hsi" / f"{name}.npy", mask)
        for line in ctx["logs"]:
            append_log(run_dir, line)


# Optional/placeholder steps for harmonization, HSI, QC
class HarmonizeStep(Step):
    name = "harmonize"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        from foodspec.harmonization import harmonize_datasets_advanced, plot_harmonization_diagnostics
        from foodspec.spectral_io import align_wavenumbers

        datasets = ctx.get("datasets")
        if datasets is None or len(datasets) == 0:
            ctx["logs"].append("[harmonize] No datasets provided; skipping.")
            return
        target_grid = self.cfg.get("target_wavenumbers")
        if len(datasets) > 1:
            curves = self.cfg.get("calibration_curves", {}) or {}
            aligned, diag = harmonize_datasets_advanced(datasets, calibration_curves=curves)
            ctx["metadata"].setdefault("harmonization", {})["diagnostics"] = diag
            # save simple mean overlay plot into figures dict
            fig = plot_harmonization_diagnostics(aligned)
            ctx["figures"]["harmonization/mean_overlay"] = fig
            ctx["logs"].append(
                "[harmonize] Advanced harmonization across "
                f"{len(datasets)} datasets; residual_std_mean={diag.get('residual_std_mean'):.4g}"
            )
        else:
            aligned = align_wavenumbers(datasets, target_grid=target_grid)
            ctx["logs"].append("[harmonize] Completed wavenumber alignment.")
        ctx["datasets"] = aligned
        ctx["data"] = aligned[0].metadata.join(
            pd.DataFrame(aligned[0].spectra, columns=[f"{wn:.4f}" for wn in aligned[0].wavenumbers])
        )
        # record per-instrument info
        instruments = []
        for ds in aligned:
            instruments.append(ds.instrument_meta if hasattr(ds, "instrument_meta") else {})
        ctx["metadata"].setdefault("harmonization", {})["instruments"] = instruments


class HSISegmentStep(Step):
    name = "hsi_segment"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        hsi = ctx.get("hsi")
        if hsi is None:
            ctx["logs"].append("[hsi_segment] No HSI dataset; skipping.")
            return
        method = self.cfg.get("method", "kmeans")
        n_clusters = self.cfg.get("n_clusters", 3)
        labels = hsi.segment(method=method, n_clusters=n_clusters)
        ctx["hsi_labels"] = labels
        ctx["figures"]["hsi/label_map"] = labels
        counts = pd.Series(labels.ravel()).value_counts().reset_index()
        counts.columns = ["label", "pixels"]
        ctx["tables"]["hsi_label_counts"] = counts
        ctx["logs"].append(f"[hsi_segment] Segmented HSI with {method}, clusters={n_clusters}.")


class HSIRoiStep(Step):
    name = "hsi_roi_to_1d"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        hsi = ctx.get("hsi")
        labels = ctx.get("hsi_labels")
        if hsi is None or labels is None:
            ctx["logs"].append("[hsi_roi_to_1d] Missing HSI or labels; skipping.")
            return
        if ctx.get("cancel"):
            ctx["logs"].append("[hsi_roi_to_1d] cancelled")
            return
        dfs = []
        roi_masks = {}
        peak_defs_cfg = self.cfg.get("peaks", [])
        peak_defs = [
            PeakDefinition(
                name=p["name"],
                column=p.get("column", p["name"]),
                wavenumber=p.get("wavenumber"),
                window=tuple(p.get("window")) if p.get("window") else None,
            )
            for p in peak_defs_cfg
        ]
        for k in np.unique(labels):
            mask = labels == k
            roi_masks[f"label_{k}"] = mask
            roi_ds = hsi.roi_spectrum(mask)
            df_peaks = roi_ds.to_peaks(peak_defs)
            df_peaks["roi_label"] = k
            dfs.append(df_peaks)
        if dfs:
            roi_df = pd.concat(dfs, ignore_index=True)
            ctx["data"] = roi_df
            ctx["tables"]["hsi_roi_peaks"] = roi_df
            ctx["hsi"].roi_masks = roi_masks
            ctx["figures"]["hsi/labels_preview"] = labels
            ctx["logs"].append("[hsi_roi_to_1d] Extracted ROI spectra to 1D table.")
            if self.cfg.get("run_rq"):
                peaks_cfg = self.cfg.get("peaks", [])
                ratios_cfg = self.cfg.get("ratios", [])
                peaks = [
                    PeakDefinition(
                        name=p["name"],
                        column=p.get("column", p["name"]),
                        wavenumber=p.get("wavenumber"),
                        window=tuple(p.get("window")) if p.get("window") else None,
                    )
                    for p in peaks_cfg
                ]
                ratios = [
                    RatioDefinition(
                        name=r["name"],
                        numerator=r["numerator"],
                        denominator=r["denominator"],
                    )
                    for r in ratios_cfg
                ]
                rq_cfg = RQConfig(
                    oil_col=self.cfg.get("oil_col", "oil_type"),
                    matrix_col=self.cfg.get("matrix_col", "matrix"),
                    heating_col=self.cfg.get("heating_col", "heating_stage"),
                    random_state=self.cfg.get("random_state", 0),
                )
                engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=rq_cfg)
                res = engine.run_all(roi_df)
                ctx["tables"]["hsi_roi_rq"] = res.stability_summary
                ctx["figures"].update(getattr(res, "figures", {}) or {})
                ctx["report"] = res.text_report
                ctx["summary"] = " ".join(res.text_report.splitlines()[:10])


class QCStep(Step):
    name = "qc_checks"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        from foodspec.validation import validate_dataset

        df = ctx.get("data")
        required = self.cfg.get("required_columns", [])
        class_col = self.cfg.get("class_col")
        diag = validate_dataset(df, required_cols=required, class_col=class_col)
        ctx["logs"].append(f"[qc_checks] warnings={diag['warnings']}")
        if diag["errors"]:
            raise ValueError(f"QC failed: {diag['errors']}")


STEP_REGISTRY = {
    PreprocessStep.name: PreprocessStep,
    RQAnalysisStep.name: RQAnalysisStep,
    OutputStep.name: OutputStep,
    HarmonizeStep.name: HarmonizeStep,
    HSISegmentStep.name: HSISegmentStep,
    HSIRoiStep.name: HSIRoiStep,
    QCStep.name: QCStep,
}


def list_available_protocols(proto_dir: Union[str, Path] = "examples/protocols") -> List[Path]:
    p = Path(proto_dir)
    if not p.exists():
        return []
    return list(p.glob("*.yml")) + list(p.glob("*.yaml")) + list(p.glob("*.json"))


def load_protocol(name: str, proto_dir: Union[str, Path] = "examples/protocols") -> ProtocolConfig:
    p = Path(proto_dir)
    path = p / name
    if not path.exists():
        # try add extension
        for ext in [".yaml", ".yml", ".json"]:
            if (p / (name + ext)).exists():
                path = p / (name + ext)
                break
    if not path.exists():
        raise FileNotFoundError(f"Protocol {name} not found in {proto_dir}")
    return ProtocolConfig.from_file(path)


# Validation helpers
def validate_protocol(cfg: ProtocolConfig, df: pd.DataFrame) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not cfg.name:
        errors.append("Protocol name missing.")
    if not cfg.version:
        warnings.append("Protocol version missing; defaulting.")
    if not cfg.steps:
        errors.append("Protocol has no steps.")
    if cfg.min_foodspec_version:
        # Best effort version check
        try:
            from foodspec import __version__ as fs_version

            if fs_version < cfg.min_foodspec_version:
                warnings.append(f"Protocol expects FoodSpec >= {cfg.min_foodspec_version}, running {fs_version}.")
        except Exception:
            warnings.append("Could not verify FoodSpec version.")
    # step type validation
    for step in cfg.steps:
        if step.get("type") not in STEP_REGISTRY:
            errors.append(f"Unknown step type: {step.get('type')}")
    # expected columns
    if cfg.expected_columns:
        for _, col in cfg.expected_columns.items():
            if col and col not in df.columns:
                errors.append(
                    f"Required column '{col}' not found. Map columns correctly or adjust protocol expected_columns."
                )
    if cfg.required_metadata:
        for m in cfg.required_metadata:
            if m not in df.columns:
                warnings.append(f"Required metadata '{m}' not found in dataset.")
    # class count
    oil_col = cfg.expected_columns.get("oil_col")
    if oil_col and oil_col in df.columns and df[oil_col].nunique(dropna=True) < 2:
        errors.append("Only one class present; add more classes/samples before running discrimination.")
    # minimal class counts
    if oil_col and oil_col in df.columns:
        min_count = df[oil_col].value_counts(dropna=True).min()
        if pd.notna(min_count) and min_count < 2:
            errors.append(
                f"Very small class count detected (min {min_count}); collect more samples or adjust protocol."
            )
        elif pd.notna(min_count) and min_count < 3:
            warnings.append(f"Small class count (min {min_count}); CV folds will be reduced automatically.")
    # constant columns
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            warnings.append(f"Column '{col}' is constant.")
    # feature/samples ratio
    num_features = df.select_dtypes(include=["number"]).shape[1]
    num_samples = len(df)
    if num_samples and num_features > 10 * num_samples:
        warnings.append(
            f"High feature-to-sample ratio ({num_features} features vs {num_samples} samples); "
            "consider feature capping or simpler normalization."
        )
    return {"errors": errors, "warnings": warnings}


class ProtocolRunner:
    def __init__(self, config: ProtocolConfig):
        self.config = config
        self._cancel = False

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ProtocolRunner":
        return cls(ProtocolConfig.from_file(path))

    def request_cancel(self):
        self._cancel = True

    def run(self, input_datasets: List[Union[pd.DataFrame, str, Path]]) -> ProtocolRunResult:
        """
        Multi-dataset aware runner.
        - Supports multiple spectral datasets (for harmonize) and HSI datasets.
        - Uses the first table as primary data for steps that expect a single df.
        """
        if not input_datasets and self.config.inputs:
            input_datasets = [Path(inp["path"]) for inp in self.config.inputs if "path" in inp]
        if not input_datasets:
            raise ValueError("No inputs provided.")

        datasets: List[SpectralDataset] = []
        hsi_list: List[HyperspectralDataset] = []
        tables: List[pd.DataFrame] = []

        def _load_one(raw_input):
            if isinstance(raw_input, HyperspectralDataset):
                hsi_list.append(raw_input)
                tables.append(raw_input.metadata.copy() if not raw_input.metadata.empty else pd.DataFrame())
                return
            if isinstance(raw_input, SpectralDataset):
                datasets.append(raw_input)
                df_spectra = pd.DataFrame(raw_input.spectra, columns=[f"{wn:.4f}" for wn in raw_input.wavenumbers])
                tables.append(pd.concat([raw_input.metadata.reset_index(drop=True), df_spectra], axis=1))
                return
            if isinstance(raw_input, (str, Path)):
                p = Path(raw_input)
                if p.suffix.lower() in {".h5", ".hdf5"}:
                    try:
                        hsi_obj = HyperspectralDataset.from_hdf5(p)
                        hsi_list.append(hsi_obj)
                        tables.append(hsi_obj.metadata.copy() if not hsi_obj.metadata.empty else pd.DataFrame())
                        return
                    except Exception:
                        spectral_obj = SpectralDataset.from_hdf5(p)
                        datasets.append(spectral_obj)
                        df_spectra = pd.DataFrame(
                            spectral_obj.spectra,
                            columns=[f"{wn:.4f}" for wn in spectral_obj.wavenumbers],
                        )
                        tables.append(pd.concat([spectral_obj.metadata.reset_index(drop=True), df_spectra], axis=1))
                        return
                tables.append(pd.read_csv(p))
                return
            tables.append(raw_input)

        for inp in input_datasets:
            _load_one(inp)

        primary_df = tables[0]
        diag = validate_protocol(self.config, primary_df)
        if diag["errors"]:
            raise ValueError("; ".join(diag["errors"]))
        ctx: Dict[str, Any] = {
            "data": primary_df,
            "logs": [],
            "metadata": {
                "protocol": self.config.name,
                "protocol_version": self.config.version,
                "min_foodspec_version": self.config.min_foodspec_version,
                "seed": self.config.seed,
                "inputs": [str(inp) for inp in input_datasets],
                "validation_strategy": self.config.validation_strategy,
            },
            "tables": {},
            "figures": {},
            "report": "",
            "summary": "",
            "run_dir": None,
            "validation": diag,
            "hsi": hsi_list[0] if hsi_list else None,
            "dataset": datasets[0] if datasets else None,
            "datasets": datasets,
            "cancel": False,
            "model_path": None,
            "registry_path": None,
        }
        start = time.time()
        # Reproducibility
        try:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
        except Exception:
            pass
        # Light guardrails
        guard_df = primary_df if isinstance(primary_df, pd.DataFrame) else None
        guard_hsi = hsi_list[0] if hsi_list else None
        if guard_df is not None:
            if guard_df.shape[0] * guard_df.shape[1] > 2_000_000:
                ctx["logs"].append(f"[warn] Large dataset ({guard_df.shape}); consider sub-sampling or dry-run first.")
            if guard_df.shape[1] > 10 * max(1, guard_df.shape[0]):
                ctx["logs"].append("[warn] High feature-to-sample ratio; RQ may auto-cap or warn.")
        if guard_hsi is not None and guard_hsi.spectra.size > 5_000_000:
            ctx["logs"].append("[warn] Large HSI cube; segmentation may be slow.")
        for step_cfg in self.config.steps:
            if self._cancel:
                ctx["logs"].append("[cancelled] User requested cancel; stopping protocol.")
                break
            step_name = step_cfg.get("type")
            step_params = step_cfg.get("params", {})
            # Auto-adjust CV folds if class counts are small
            if step_name == "rq_analysis" and isinstance(primary_df, pd.DataFrame):
                oil_col = self.config.expected_columns.get("oil_col", "oil_type")
                if oil_col in primary_df.columns:
                    min_count = primary_df[oil_col].value_counts(dropna=True).min()
                    default_splits = step_params.get("n_splits", 5)
                    if pd.notna(min_count) and min_count < default_splits:
                        new_splits = max(2, int(min_count))
                        step_params["n_splits"] = new_splits
                        ctx["logs"].append(f"[auto] Reduced CV folds to {new_splits} due to small class counts.")
                        ctx["metadata"].setdefault("auto_adjustments", {})["cv_folds"] = new_splits
            step_cls = STEP_REGISTRY.get(step_name)
            if not step_cls:
                ctx["logs"].append(f"[skip] Unknown step {step_name}")
                continue
            step = step_cls(step_params)
            step.run(ctx)
            if self._cancel:
                ctx["logs"].append(f"[cancelled] Stopped after step {step_name}.")
                break
        ctx["metadata"]["duration_sec"] = time.time() - start
        return ProtocolRunResult(
            run_dir=ctx.get("run_dir"),
            logs=ctx["logs"],
            metadata=ctx["metadata"],
            tables=ctx["tables"],
            figures=ctx["figures"],
            report=ctx["report"],
            summary=ctx["summary"],
        )

    def save_outputs(self, result: ProtocolRunResult, output_dir: Union[str, Path]):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.txt").write_text(result.report, encoding="utf-8")
        meta = result.metadata.copy()
        meta["protocol_version"] = self.config.version
        meta["min_foodspec_version"] = self.config.min_foodspec_version
        meta["logs"] = result.logs
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        save_tables(out_dir, result.tables)
        save_figures(out_dir, result.figures)


# Example protocol (JSON) for edible oils Raman classification
EXAMPLE_PROTOCOL = {
    "name": "EdibleOil_Raman_Classification_v1",
    "description": "Raman preprocessing + RQ analysis for edible oils",
    "seed": 0,
    "steps": [
        {
            "type": "preprocess",
            "params": {
                "baseline_lambda": 1e5,
                "baseline_p": 0.01,
                "smooth_window": 9,
                "smooth_polyorder": 3,
                "normalization": "reference",
                "reference_wavenumber": 2720.0,
                "peaks": [
                    {"name": "I_1742", "wavenumber": 1742},
                    {"name": "I_1652", "wavenumber": 1652},
                    {"name": "I_1434", "wavenumber": 1434},
                    {"name": "I_1296", "wavenumber": 1296},
                    {"name": "I_1259", "wavenumber": 1259},
                    {"name": "I_2720", "wavenumber": 2720},
                ],
            },
        },
        {
            "type": "rq_analysis",
            "params": {
                "oil_col": "oil_type",
                "matrix_col": "matrix",
                "heating_col": "heating_stage",
                "random_state": 0,
                "n_splits": 5,
                "normalization_modes": ["reference", "vector", "area"],
                "ratios": [
                    {"name": "1742/2720", "numerator": "I_1742", "denominator": "I_2720"},
                    {"name": "1652/2720", "numerator": "I_1652", "denominator": "I_2720"},
                    {"name": "1434/2720", "numerator": "I_1434", "denominator": "I_2720"},
                    {"name": "1259/2720", "numerator": "I_1259", "denominator": "I_2720"},
                    {"name": "1296/2720", "numerator": "I_1296", "denominator": "I_2720"},
                ],
            },
        },
        {
            "type": "output",
            "params": {
                "output_dir": "protocol_runs",
            },
        },
    ],
}
