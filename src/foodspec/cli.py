"""Command-line interface for foodspec."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.pipeline import Pipeline

from foodspec import __version__
from foodspec.apps.dairy import run_dairy_authentication_workflow
from foodspec.apps.heating import run_heating_degradation_analysis
from foodspec.apps.meat import run_meat_authentication_workflow
from foodspec.apps.microbial import run_microbial_detection_workflow
from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.apps.protocol_validation import run_protocol_benchmarks
from foodspec.apps.qc import apply_qc_model, train_qc_model
from foodspec.chemometrics.mixture import nnls_mixture
from foodspec.config import load_config, merge_cli_overrides
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.core.hyperspectral import HyperSpectralCube
from foodspec.core.api import FoodSpec
from foodspec.data.libraries import load_library
from foodspec.io import create_library, load_csv_spectra
from foodspec.io.exporters import to_hdf5
from foodspec.io.loaders import load_folder
from foodspec.logging_utils import get_logger, log_run_metadata
from foodspec.model_registry import load_model as registry_load_model
from foodspec.model_registry import save_model as registry_save_model
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.cropping import RangeCropper
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.reporting import (
    create_report_folder,
    save_figure,
    write_json,
    write_markdown_report,
    write_metrics_csv,
    write_summary_json,
)
from foodspec.features.specs import FeatureSpec
from foodspec.repro.experiment import ExperimentEngine, ExperimentConfig
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.viz.heating import plot_ratio_vs_time
from foodspec.viz.hyperspectral import plot_hyperspectral_intensity_map
from foodspec.viz.report import render_html_report_oil_auth

app = typer.Typer(help="foodspec command-line interface")
logger = get_logger(__name__)


def _create_report_dir(workflow_name: str, base_dir: Path) -> Path:
    """Create timestamped report directory."""

    return create_report_folder(base_dir, workflow_name)


def _to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas objects to JSON-serializable equivalents."""

    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series,)):
        return {k: _to_serializable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, (pd.DataFrame,)):
        return [{k: _to_serializable(v) for k, v in row.items()} for row in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def _detect_optional_extras() -> list[str]:
    """Detect installed optional extras such as deep learning or gradient boosting."""

    optional_pkgs = ["tensorflow", "torch", "xgboost", "lightgbm"]
    installed = []
    for pkg in optional_pkgs:
        if importlib.util.find_spec(pkg) is not None:
            installed.append(pkg)
    return installed


def _apply_seeds(seeds: dict[str, Any]) -> None:
    """Set random seeds across common libraries if provided."""

    if not seeds:
        return
    if "python_random_seed" in seeds:
        import random as _random

        _random.seed(seeds["python_random_seed"])
    if "numpy_seed" in seeds:
        np.random.seed(int(seeds["numpy_seed"]))
    try:
        import torch  # type: ignore

        if "torch_seed" in seeds:
            torch.manual_seed(int(seeds["torch_seed"]))
    except Exception:
        pass


def _build_feature_specs(raw_specs: list[dict[str, Any]]) -> list[FeatureSpec]:
    """Construct FeatureSpec objects from YAML dictionaries."""

    specs: list[FeatureSpec] = []
    for spec in raw_specs:
        specs.append(
            FeatureSpec(
                name=spec.get("name", "feature"),
                ftype=spec.get("ftype", "band"),
                regions=spec.get("regions"),
                formula=spec.get("formula"),
                label=spec.get("label"),
                description=spec.get("description"),
                citation=spec.get("citation"),
                constraints=spec.get("constraints", {}),
                params=spec.get("params", {}),
            )
        )
    return specs


def _write_oil_report(
    result,
    spectra: FoodSpectrumSet,
    label_column: str,
    output_report: Path,
    classifier_name: str,
    run_metadata: Optional[dict] = None,
) -> Path:
    """Write oil authentication report folder and return its path."""

    report_dir = _create_report_dir("oil_auth", output_report.parent)
    # Summary
    summary = {
        "workflow": "oil_auth",
        "n_samples": len(spectra),
        "class_labels": list(result.class_labels),
        "classifier_name": classifier_name,
    }
    # Metrics CSV
    write_metrics_csv(report_dir, "metrics", result.cv_metrics)
    summary["cv_metrics_mean"] = _to_serializable(result.cv_metrics.select_dtypes(include=[np.number]).mean())
    # Confusion matrix
    if result.confusion_matrix is not None:
        fig, ax = plt.subplots()
        plot_confusion_matrix(result.confusion_matrix, class_names=result.class_labels, ax=ax)
        save_figure(report_dir, "confusion_matrix", fig)
        plt.close(fig)
    # Feature importances
    if result.feature_importances is not None:
        write_metrics_csv(report_dir, "feature_importances", result.feature_importances.to_frame("importance"))
    if run_metadata is not None:
        write_json(report_dir / "run_metadata.json", run_metadata)
    # Markdown report
    metrics_text = json.dumps(summary.get("cv_metrics_mean", {}), indent=2)
    sections = {
        "Description": "Oil authentication workflow (baseline, smoothing, normalization, peaks/ratios, classifier).",
        "Key metrics": f"```\n{metrics_text}\n```",
        "Figures": "See confusion_matrix.png (if available).",
    }
    write_markdown_report(report_dir / "report.md", title="Oil Authentication", sections=sections)
    # Summary JSON
    write_summary_json(report_dir, _to_serializable(summary))
    return report_dir


def _write_heating_report(result, output_dir: Path, time_column: str) -> Path:
    """Write heating workflow report."""

    report_dir = _create_report_dir("heating", output_dir)
    write_metrics_csv(report_dir, "ratios", result.key_ratios)
    # Trend model metrics
    rows = []
    for name, model in result.trend_models.items():
        if name == "by_oil_type":
            continue
        if hasattr(model, "coef_"):
            rows.append(
                {
                    "ratio": name,
                    "slope": float(model.coef_.ravel()[0]),
                    "intercept": float(model.intercept_.ravel()[0]),
                }
            )
    if rows:
        write_metrics_csv(report_dir, "trend_models", pd.DataFrame(rows))
    # ANOVA
    if result.anova_results is not None:
        write_metrics_csv(report_dir, "anova_results", result.anova_results)
    # Plot first ratio vs time
    if not result.key_ratios.empty:
        ratio_col = (
            "ratio_1655_1742" if "ratio_1655_1742" in result.key_ratios.columns else result.key_ratios.columns[0]
        )
        fig, ax = plt.subplots()
        plot_ratio_vs_time(
            result.time_variable,
            result.key_ratios[ratio_col],
            model=result.trend_models.get(ratio_col),
            ax=ax,
        )
        save_figure(report_dir, "ratio_vs_time", fig)
        plt.close(fig)
    summary = {
        "workflow": "heating",
        "n_samples": len(result.key_ratios),
        "ratios": list(result.key_ratios.columns),
        "anova_present": result.anova_results is not None,
        "time_column": time_column,
    }
    write_summary_json(report_dir, _to_serializable(summary))
    return report_dir


def _write_qc_report(qc_result, output_dir: Path, model_type: str, threshold: float) -> Path:
    """Write QC workflow report."""

    report_dir = _create_report_dir("qc", output_dir)
    scores_df = qc_result.metadata.copy()
    scores_df["score"] = qc_result.scores
    scores_df["label_pred"] = qc_result.labels_pred
    write_metrics_csv(report_dir, "scores", scores_df)

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(qc_result.scores, bins=20, alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", label="threshold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    save_figure(report_dir, "scores_hist", fig)
    plt.close(fig)

    summary = {
        "workflow": "qc",
        "model_type": model_type,
        "threshold": float(threshold),
        "counts": qc_result.labels_pred.value_counts().to_dict(),
    }
    write_summary_json(report_dir, _to_serializable(summary))
    return report_dir


def _write_domain_report(result, output_dir: Path, domain: str, classifier_name: str) -> Path:
    """Write domain workflow report."""

    report_dir = _create_report_dir(domain, output_dir)
    write_metrics_csv(report_dir, "cv_metrics", result.cv_metrics)
    if result.confusion_matrix is not None:
        cm_df = pd.DataFrame(result.confusion_matrix, index=result.class_labels, columns=result.class_labels)
        write_metrics_csv(report_dir, "confusion_matrix", cm_df)
        fig, ax = plt.subplots()
        plot_confusion_matrix(result.confusion_matrix, class_names=result.class_labels, ax=ax)
        save_figure(report_dir, f"confusion_matrix_{domain}", fig)
        plt.close(fig)
    summary = {
        "workflow": domain,
        "classifier_name": classifier_name,
        "n_classes": len(result.class_labels),
    }
    write_summary_json(report_dir, _to_serializable(summary))
    return report_dir


def _write_mixture_report(
    spectrum_index: int,
    coeffs: np.ndarray,
    residual: float,
    pure_labels: Optional[pd.Series],
    output_dir: Path,
) -> Path:
    """Write mixture decomposition report."""

    report_dir = _create_report_dir("mixture", output_dir)
    labels = list(pure_labels) if pure_labels is not None else [f"comp_{i}" for i in range(len(coeffs))]
    df = pd.DataFrame({"component": labels, "coefficient": coeffs})
    write_metrics_csv(report_dir, "coefficients", df)
    summary = {
        "workflow": "mixture",
        "spectrum_index": spectrum_index,
        "residual_norm": float(residual),
        "n_components": len(coeffs),
        "coefficients": dict(zip(labels, map(float, coeffs))),
    }
    write_summary_json(report_dir, _to_serializable(summary))
    return report_dir


def _write_hyperspectral_report(
    cube: HyperSpectralCube,
    target_wavenumber: float,
    window: float,
    output_dir: Path,
) -> Path:
    """Write hyperspectral report."""

    report_dir = _create_report_dir("hyperspectral", output_dir)
    mask = np.abs(cube.wavenumbers - target_wavenumber) <= window
    if not np.any(mask):
        raise typer.BadParameter("No wavenumbers within specified window.")
    intensity_map = cube.cube[:, :, mask].mean(axis=2)
    fig, ax = plt.subplots()
    plot_hyperspectral_intensity_map(cube, target_wavenumber=target_wavenumber, window=window, ax=ax)
    save_figure(report_dir, "intensity_map", fig)
    plt.close(fig)

    summary = {
        "workflow": "hyperspectral",
        "height": int(cube.image_shape[0]),
        "width": int(cube.image_shape[1]),
        "n_points": int(cube.wavenumbers.shape[0]),
        "target_wavenumber": float(target_wavenumber),
        "window": float(window),
        "intensity_stats": {
            "min": float(np.min(intensity_map)),
            "max": float(np.max(intensity_map)),
            "mean": float(np.mean(intensity_map)),
        },
    }
    write_summary_json(report_dir, _to_serializable(summary))
    return report_dir


class _CropperWrapper(RangeCropper):
    """Pipeline-friendly wrapper over RangeCropper that stores axis."""

    def __init__(self, wavenumbers: np.ndarray, min_wn: float, max_wn: float):
        self.wavenumbers_full = np.asarray(wavenumbers, dtype=float)
        super().__init__(min_wn=min_wn, max_wn=max_wn)
        mask = (self.wavenumbers_full >= self.min_wn) & (self.wavenumbers_full <= self.max_wn)
        if not np.any(mask):
            raise ValueError("Cropping mask is empty.")
        self.mask_ = mask
        self.wavenumbers_ = self.wavenumbers_full[mask]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.wavenumbers_full.shape[0]:
            raise ValueError("Input X columns must match length of original wavenumbers.")
        return X[:, self.mask_]


def _default_preprocess_pipeline(wavenumbers: np.ndarray, min_wn: float, max_wn: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("als", ALSBaseline(lambda_=1e5, p=0.01, max_iter=10)),
            ("savgol", SavitzkyGolaySmoother(window_length=9, polyorder=3)),
            ("norm", VectorNormalizer(norm="l2")),
            ("crop", _CropperWrapper(wavenumbers=wavenumbers, min_wn=min_wn, max_wn=max_wn)),
        ]
    )


@app.command("preprocess")
def preprocess(
    input_folder: str = typer.Argument(..., help="Folder containing spectra text files."),
    metadata_csv: Optional[str] = typer.Option(None, help="Optional metadata CSV with sample_id."),
    output_hdf5: str = typer.Argument(..., help="Output HDF5 path."),
    modality: str = typer.Option("raman", help="Spectroscopy modality."),
    min_wn: float = typer.Option(600.0, help="Minimum wavenumber for cropping."),
    max_wn: float = typer.Option(1800.0, help="Maximum wavenumber for cropping."),
):
    """Load spectra, apply default preprocessing, and save to HDF5."""

    ds = load_folder(
        folder=input_folder,
        metadata_csv=metadata_csv,
        modality=modality,
    )
    pipe = _default_preprocess_pipeline(ds.wavenumbers, min_wn=min_wn, max_wn=max_wn)
    x_proc = pipe.fit_transform(ds.x)
    cropper = pipe.named_steps["crop"]
    ds_out = FoodSpectrumSet(
        x=x_proc,
        wavenumbers=cropper.wavenumbers_,
        metadata=ds.metadata.copy(),
        modality=ds.modality,
    )
    to_hdf5(ds_out, output_hdf5)
    typer.echo(f"Preprocessed spectra saved to {output_hdf5}")


@app.command("oil-auth")
def oil_auth(
    input_hdf5: str = typer.Argument(..., help="Input HDF5 file with spectra."),
    label_column: str = typer.Option("oil_type", help="Metadata column for class labels."),
    cv_splits: int = typer.Option(5, help="CV splits for classifier."),
    output_report: str = typer.Option("oil_auth_report.html", help="Output HTML report path."),
    save_model_path: Optional[str] = typer.Option(
        None, "--save-model", help="Optional base path to save the trained model (without extension)."
    ),
    model_version: str = typer.Option(__version__, help="Model version tag."),
    config: Optional[str] = typer.Option(None, "--config", help="Optional YAML/JSON config file."),
):
    """Run oil authentication workflow and save HTML report."""

    _run_meta = log_run_metadata(logger, {"command": "oil-auth"})
    base_cfg = {
        "input_hdf5": input_hdf5,
        "label_column": label_column,
        "output_report": output_report,
        "cv_splits": cv_splits,
    }
    cfg = load_config(config) if config else base_cfg
    cfg = merge_cli_overrides(cfg, base_cfg)

    ds = load_library(cfg["input_hdf5"])
    result = run_oil_authentication_workflow(
        spectra=ds,
        label_column=cfg.get("label_column", label_column),
        cv_splits=cfg.get("cv_splits", cv_splits),
    )
    render_html_report_oil_auth(result, cfg.get("output_report", output_report))
    classifier_name = result.pipeline.named_steps.get("clf").__class__.__name__ if result.pipeline else "unknown"
    report_dir = _write_oil_report(
        result,
        ds,
        label_column=label_column,
        output_report=Path(output_report),
        classifier_name=classifier_name,
        run_metadata=_run_meta,
    )
    typer.echo(f"Report folder: {report_dir}")
    if save_model_path is not None:
        name = f"oil_{classifier_name.lower()}"
        registry_save_model(
            result.pipeline,
            save_model_path,
            name=name,
            version=model_version,
            foodspec_version=__version__,
            extra={
                "command": "oil-auth",
                "label_column": label_column,
                "classifier_name": classifier_name,
                "class_labels": list(result.class_labels),
            },
        )
        typer.echo(f"Model saved: {save_model_path}.joblib / {save_model_path}.json")
    typer.echo(f"HTML report written to {output_report}")


@app.command("heating")
def heating_command(
    input_hdf5: str = typer.Argument(..., help="Preprocessed spectra HDF5."),
    time_column: str = typer.Option("heating_time", help="Metadata column for heating time."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """Run heating degradation workflow and write report folder."""

    _run_meta = log_run_metadata(logger, {"command": "heating"})
    ds = load_library(input_hdf5)
    result = run_heating_degradation_analysis(ds, time_column=time_column)
    report_dir = _write_heating_report(result, Path(output_dir), time_column=time_column)
    typer.echo(f"Heating report: {report_dir}")


@app.command("qc")
def qc_command(
    input_hdf5: str = typer.Argument(..., help="Preprocessed spectra HDF5."),
    model_type: str = typer.Option("oneclass_svm", help="QC model type: oneclass_svm or isolation_forest."),
    label_column: Optional[str] = typer.Option(None, help="Optional label column for inspection."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """Run QC/novelty detection and write report."""

    ds = load_library(input_hdf5)
    model = train_qc_model(ds, train_mask=None, model_type=model_type)
    qc_result = apply_qc_model(ds, model=model, metadata=ds.metadata)
    report_dir = _write_qc_report(qc_result, Path(output_dir), model_type=model_type, threshold=qc_result.threshold)
    typer.echo(f"QC report: {report_dir}")


@app.command("domains")
def domains_command(
    input_hdf5: str = typer.Argument(..., help="Preprocessed spectra HDF5."),
    domain: str = typer.Option(..., "--type", help="Domain type: dairy, meat, microbial."),
    label_column: str = typer.Option("label", help="Metadata column with class labels."),
    classifier_name: str = typer.Option("rf", help="Classifier name."),
    cv_splits: int = typer.Option(5, help="Number of CV splits."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
    save_model_path: Optional[str] = typer.Option(
        None, "--save-model", help="Optional base path to save the trained model (without extension)."
    ),
    model_version: str = typer.Option(__version__, help="Model version tag."),
):
    """Run domain-specific authentication templates and write report."""

    ds = load_library(input_hdf5)
    domain_lower = domain.lower()
    if domain_lower == "dairy":
        result = run_dairy_authentication_workflow(
            ds, label_column=label_column, classifier_name=classifier_name, cv_splits=cv_splits
        )
    elif domain_lower == "meat":
        result = run_meat_authentication_workflow(
            ds, label_column=label_column, classifier_name=classifier_name, cv_splits=cv_splits
        )
    elif domain_lower == "microbial":
        result = run_microbial_detection_workflow(
            ds, label_column=label_column, classifier_name=classifier_name, cv_splits=cv_splits
        )
    else:
        raise typer.BadParameter("domain must be one of: dairy, meat, microbial.")

    report_dir = _write_domain_report(result, Path(output_dir), domain=domain_lower, classifier_name=classifier_name)

    if save_model_path is not None:
        name = f"{domain_lower}_{classifier_name.lower()}"
        registry_save_model(
            result.pipeline,
            save_model_path,
            name=name,
            version=model_version,
            foodspec_version=__version__,
            extra={
                "command": "domains",
                "domain": domain_lower,
                "classifier_name": classifier_name,
                "label_column": label_column,
                "cv_splits": cv_splits,
                "class_labels": list(result.class_labels),
            },
        )
        typer.echo(f"Model saved: {save_model_path}.joblib / {save_model_path}.json")

    typer.echo(f"{domain} report: {report_dir}")


@app.command("mixture")
def mixture_command(
    input_hdf5: str = typer.Argument(..., help="Preprocessed spectra HDF5."),
    pure_hdf5: str = typer.Option(..., help="HDF5 with pure component spectra."),
    spectrum_index: int = typer.Option(0, help="Index of spectrum in input file to decompose."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """Perform NNLS mixture analysis on a single spectrum and write report."""

    spectra = load_library(input_hdf5)
    pure = load_library(pure_hdf5)
    if pure.wavenumbers.shape != spectra.wavenumbers.shape or not np.allclose(pure.wavenumbers, spectra.wavenumbers):
        raise typer.BadParameter("Pure and input wavenumbers must match.")
    if spectrum_index < 0 or spectrum_index >= len(spectra):
        raise typer.BadParameter("spectrum_index out of range.")

    spectrum = spectra.x[spectrum_index]
    pure_mat = pure.x.T  # n_points x n_components
    coeffs, res = nnls_mixture(spectrum, pure_mat)
    reconstructed = pure_mat @ coeffs

    fig, ax = plt.subplots()
    ax.plot(spectra.wavenumbers, spectrum, label="original")
    ax.plot(spectra.wavenumbers, reconstructed, label="reconstructed", linestyle="--")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")
    ax.legend()
    report_dir = _write_mixture_report(
        spectrum_index=spectrum_index,
        coeffs=coeffs,
        residual=res,
        pure_labels=pure.metadata["sample_id"] if "sample_id" in pure.metadata.columns else None,
        output_dir=Path(output_dir),
    )
    save_figure(report_dir, "mixture_fit", fig)
    plt.close(fig)
    typer.echo(f"Mixture report: {report_dir}")


@app.command("hyperspectral")
def hyperspectral_command(
    input_hdf5: str = typer.Argument(..., help="Flattened pixel spectra HDF5."),
    height: int = typer.Option(..., help="Image height in pixels."),
    width: int = typer.Option(..., help="Image width in pixels."),
    target_wavenumber: float = typer.Option(1655.0, help="Target wavenumber for intensity map."),
    window: float = typer.Option(5.0, help="Integration window."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """Create hyperspectral intensity map from flattened spectra."""

    ds = load_library(input_hdf5)
    cube = HyperSpectralCube.from_spectrum_set(ds, image_shape=(height, width))
    report_dir = _write_hyperspectral_report(
        cube=cube, target_wavenumber=target_wavenumber, window=window, output_dir=Path(output_dir)
    )
    typer.echo(f"Hyperspectral report: {report_dir}")


@app.command("run-exp")
def run_experiment(
    exp_path: Path = typer.Argument(..., help="Path to exp.yml experiment file."),
    output_dir: Optional[Path] = typer.Option(None, help="Override base output directory."),
    dry_run: bool = typer.Option(False, help="Only validate and summarize the experiment config."),
):
    """Execute an experiment defined in exp.yml (reproducible, single-command pipeline)."""

    engine = ExperimentEngine.from_yaml(exp_path)
    cfg: ExperimentConfig = engine.config
    record = cfg.build_run_record()

    typer.echo(cfg.summary())
    typer.echo(f"config_hash={cfg.config_hash} dataset_hash={record.dataset_hash}")
    if dry_run:
        return

    _apply_seeds(cfg.seeds)

    # Load data and initialize workflow
    fs = FoodSpec(cfg.dataset.path, modality=cfg.dataset.modality)
    fs.bundle.run_record = record  # Attach provenance built from exp.yml

    # QC
    if cfg.qc:
        qc_threshold = cfg.qc.get("threshold")
        if qc_threshold is None:
            qc_threshold = (cfg.qc.get("thresholds") or {}).get("outlier_rate", 0.5)
        fs.qc(method=cfg.qc.get("method", "robust_z"), threshold=qc_threshold)

    # Preprocessing
    pre_cfg = cfg.preprocessing or {}
    pre_args = {k: v for k, v in pre_cfg.items() if k != "preset"}
    fs.preprocess(pre_cfg.get("preset", "auto"), **pre_args)

    # Features
    feat_cfg = cfg.features or {}
    feat_preset = feat_cfg.get("preset", "standard")
    feat_specs = _build_feature_specs(feat_cfg.get("specs", [])) if feat_preset == "specs" else None
    fs.features(feat_preset, specs=feat_specs)

    # Modeling
    mod_cfg = cfg.modeling or {}
    suite = mod_cfg.get("suite") or []
    first_model = suite[0] if suite else {}
    algorithm = first_model.get("algorithm", mod_cfg.get("algorithm", "rf"))
    label_column = mod_cfg.get("label_column") or cfg.dataset.schema.get("label_column", "label")
    cv_folds = first_model.get("cv_folds", mod_cfg.get("cv_folds", 5))
    params = first_model.get("params", {})
    fs.train(algorithm=algorithm, label_column=label_column, cv_folds=cv_folds, **params)

    # Export
    base_dir = output_dir or cfg.outputs.get("base_dir") or Path("foodspec_runs") / record.run_id
    out_path = fs.export(base_dir)
    typer.echo(f"Experiment complete. Outputs: {out_path}")


@app.command("about")
def about() -> None:
    """Print version and environment information for foodspec."""

    extras = _detect_optional_extras()
    typer.echo(f"foodspec version: {__version__}")
    typer.echo(f"Python version: {sys.version.split()[0]}")
    typer.echo(f"Optional extras detected: {', '.join(extras) if extras else 'none'}")
    typer.echo("Documentation: https://github.com/your-org/foodspec#documentation")
    typer.echo("Description: foodspec is a headless, research-grade toolkit for Raman/FTIR in food science.")


@app.command("csv-to-library")
def csv_to_library(
    csv_path: str = typer.Argument(..., help="Input CSV file with spectra."),
    output_hdf5: str = typer.Argument(..., help="Output HDF5 library path (will be created or overwritten)."),
    format: str = typer.Option(
        "wide",
        "--format",
        help="CSV layout: 'wide' (one column per spectrum) or 'long' (tidy format).",
        case_sensitive=False,
    ),
    modality: str = typer.Option(
        "raman",
        "--modality",
        help="Spectroscopy modality tag (e.g. 'raman', 'ftir').",
    ),
    wavenumber_column: str = typer.Option(
        "wavenumber",
        "--wavenumber-column",
        help="Name of the wavenumber column.",
    ),
    sample_id_column: str = typer.Option(
        "sample_id",
        "--sample-id-column",
        help="For 'long' format: sample identifier column.",
    ),
    intensity_column: str = typer.Option(
        "intensity",
        "--intensity-column",
        help="For 'long' format: intensity column.",
    ),
    label_column: str = typer.Option(
        "",
        "--label-column",
        help="Optional label column name (e.g. oil_type).",
    ),
):
    """
    Convert a CSV file of spectra into an HDF5 library usable by foodspec workflows.
    """

    label_column = label_column or None
    logger.info("Loading CSV spectra from %s", csv_path)
    ds = load_csv_spectra(
        csv_path=csv_path,
        format=format,
        wavenumber_column=wavenumber_column,
        sample_id_column=sample_id_column,
        intensity_column=intensity_column,
        label_column=label_column,
        modality=modality,
    )

    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving HDF5 library to %s", output_hdf5)
    create_library(path=output_hdf5, spectra=ds)
    logger.info("Done. Library contains %s spectra.", len(ds))


@app.command("model-info")
def model_info_command(
    path: str = typer.Argument(..., help="Base path of saved model (without extension)."),
):
    """Inspect saved model metadata."""

    model_base = Path(path)
    joblib_path = model_base.with_suffix(".joblib")
    json_path = model_base.with_suffix(".json")
    if not joblib_path.exists() or not json_path.exists():
        typer.echo("Model files not found (expected .joblib and .json).", err=True)
        raise typer.Exit(code=1)
    try:
        _, meta = registry_load_model(path)
    except Exception as exc:  # pragma: no cover - defensive
        typer.echo(f"Failed to load model metadata: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Name: {meta.name}")
    typer.echo(f"Version: {meta.version}")
    typer.echo(f"Foodspec version: {meta.foodspec_version}")
    typer.echo(f"Created at: {meta.created_at}")
    typer.echo("Extra:")
    typer.echo(json.dumps(meta.extra, indent=2))


@app.command("protocol-benchmarks")
def protocol_benchmarks(
    output_dir: str = typer.Option("./protocol_benchmarks", help="Directory to write benchmark metrics."),
    random_state: int = typer.Option(42, help="Random seed."),
    config: Optional[str] = typer.Option(None, "--config", help="Optional YAML/JSON config."),
):
    """Run protocol benchmarks on public datasets and save reports."""

    base_cfg = {"output_dir": output_dir, "random_state": random_state}
    cfg = load_config(config) if config else base_cfg
    cfg = merge_cli_overrides(cfg, base_cfg)

    out_path = Path(cfg["output_dir"])
    run_meta = log_run_metadata(logger, {"command": "protocol-benchmarks"})
    summary = run_protocol_benchmarks(out_path, random_state=cfg.get("random_state", random_state))
    # write run metadata alongside metrics
    meta_path = out_path / "run_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    typer.echo("Protocol benchmarks summary:")
    typer.echo(json.dumps(summary, indent=2))




def main():
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
