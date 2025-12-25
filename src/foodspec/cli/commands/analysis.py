"""Analysis commands: domain-specific workflows (oil, heating, mixture, hyperspectral, aging, shelf-life)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from foodspec import __version__
from foodspec.apps.dairy import run_dairy_authentication_workflow
from foodspec.apps.heating import run_heating_degradation_analysis
from foodspec.apps.meat import run_meat_authentication_workflow
from foodspec.apps.microbial import run_microbial_detection_workflow
from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.chemometrics.mixture import nnls_mixture
from foodspec.config import load_config, merge_cli_overrides
from foodspec.core.hyperspectral import HyperSpectralCube
from foodspec.core.time import TimeSpectrumSet
from foodspec.data.libraries import load_library
from foodspec.logging_utils import get_logger, log_run_metadata
from foodspec.model_registry import save_model as registry_save_model
from foodspec.reporting import (
    create_report_folder,
    save_figure,
    write_metrics_csv,
    write_summary_json,
)
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.viz.heating import plot_ratio_vs_time
from foodspec.viz.report import render_html_report_oil_auth
from foodspec.workflows.aging import compute_degradation_trajectories
from foodspec.workflows.shelf_life import estimate_remaining_shelf_life

logger = get_logger(__name__)

analysis_app = typer.Typer(help="Domain-specific analysis commands")


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


def _write_oil_report(
    result, spectra, label_column: str, output_report: Path, classifier_name: str, run_metadata: Optional[dict] = None
) -> Path:
    """Write oil authentication report folder and return its path."""
    report_dir = create_report_folder(output_report.parent, "oil_auth")
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
        from foodspec.reporting import write_json

        write_json(report_dir / "run_metadata.json", run_metadata)
    # Markdown report
    import json

    metrics_text = json.dumps(summary.get("cv_metrics_mean", {}), indent=2)
    sections = {
        "Description": "Oil authentication workflow (baseline, smoothing, normalization, peaks/ratios, classifier).",
        "Key metrics": f"```\n{metrics_text}\n```",
        "Figures": "See confusion_matrix.png (if available).",
    }
    from foodspec.reporting import write_markdown_report

    write_markdown_report(report_dir / "report.md", title="Oil Authentication", sections=sections)
    # Summary JSON
    write_summary_json(report_dir, _to_serializable(summary))
    return report_dir


def _write_heating_report(result, output_dir: Path, time_column: str) -> Path:
    """Write heating workflow report."""
    report_dir = create_report_folder(output_dir, "heating")
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


def _write_domain_report(result, output_dir: Path, domain: str, classifier_name: str) -> Path:
    """Write domain workflow report."""
    report_dir = create_report_folder(output_dir, domain)
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
    spectrum_index: int, coeffs: np.ndarray, residual: float, pure_labels: Optional[pd.Series], output_dir: Path
) -> Path:
    """Write mixture decomposition report."""
    report_dir = create_report_folder(output_dir, "mixture")
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
    cube: HyperSpectralCube, target_wavenumber: float, window: float, output_dir: Path
) -> Path:
    """Write hyperspectral report."""
    report_dir = create_report_folder(output_dir, "hyperspectral")
    mask = np.abs(cube.wavenumbers - target_wavenumber) <= window
    if not np.any(mask):
        raise typer.BadParameter("No wavenumbers within specified window.")
    intensity_map = cube.cube[:, :, mask].mean(axis=2)
    fig, ax = plt.subplots()
    from foodspec.viz.hyperspectral import plot_hyperspectral_intensity_map

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


@analysis_app.command("oil-auth")
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


@analysis_app.command("heating")
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


@analysis_app.command("domains")
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


@analysis_app.command("mixture")
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


@analysis_app.command("hyperspectral")
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


@analysis_app.command("aging")
def aging_command(
    input_hdf5: str = typer.Argument(..., help="Input HDF5 library with spectra and metadata."),
    value_col: str = typer.Option(..., help="Metadata column to model over time (e.g., a ratio/feature)."),
    method: str = typer.Option("linear", help="Trajectory model: linear or spline."),
    time_col: Optional[str] = typer.Option(None, help="Time column name in metadata."),
    entity_col: Optional[str] = typer.Option(None, help="Entity identifier column (e.g., sample_id/batch_id)."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """Model degradation trajectories and stage classification.

    Computes per-entity trajectories for a numeric value over time, returns slope/acceleration
    metrics and stage labels (early/mid/late). Writes CSVs and a sample fit figure.
    """
    ds = load_library(input_hdf5)
    ts = TimeSpectrumSet(
        x=ds.x,
        wavenumbers=ds.wavenumbers,
        metadata=ds.metadata,
        modality=ds.modality,
        time_col=time_col,
        entity_col=entity_col,
    )
    result = compute_degradation_trajectories(ts, value_col=value_col, method=method)  # type: ignore[arg-type]
    report_dir = create_report_folder(Path(output_dir), "aging")
    write_metrics_csv(report_dir, "aging_metrics", result.metrics)
    write_metrics_csv(report_dir, "stages", result.stages)
    # Plot first entity fit for quick inspection
    if result.fits:
        ent = sorted(result.fits.keys())[0]
        fit = result.fits[ent]
        fig, ax = plt.subplots()
        ax.plot(fit.times, fit.values, "o", label=f"{ent} observed")
        ax.plot(fit.times, fit.fitted, "-", label=f"{fit.method} fit")
        ax.set_xlabel("Time")
        ax.set_ylabel(value_col)
        ax.legend()
        save_figure(report_dir, f"fit_{ent}", fig)
        plt.close(fig)
    write_summary_json(
        report_dir,
        {
            "workflow": "aging",
            "n_entities": int(result.metrics.shape[0]),
            "value_col": value_col,
            "method": method,
        },
    )
    typer.echo(f"Aging report: {report_dir}")


@analysis_app.command("shelf-life")
def shelf_life_command(
    input_hdf5: str = typer.Argument(..., help="Input HDF5 library with spectra and metadata."),
    value_col: str = typer.Option(..., help="Numeric metric regressed vs time (e.g., degradation index)."),
    threshold: float = typer.Option(..., help="Decision threshold to estimate remaining time to reach."),
    time_col: Optional[str] = typer.Option(None, help="Time column name in metadata."),
    entity_col: Optional[str] = typer.Option(None, help="Entity identifier column (e.g., sample_id/batch_id)."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """Estimate remaining shelf-life per entity with confidence intervals.

    Fits OLS y~t per entity and solves for t* where y crosses the threshold. Reports t*, CI.
    """
    ds = load_library(input_hdf5)
    ts = TimeSpectrumSet(
        x=ds.x,
        wavenumbers=ds.wavenumbers,
        metadata=ds.metadata,
        modality=ds.modality,
        time_col=time_col,
        entity_col=entity_col,
    )
    df = estimate_remaining_shelf_life(ts, value_col=value_col, threshold=threshold)
    report_dir = create_report_folder(Path(output_dir), "shelf_life")
    write_metrics_csv(report_dir, "shelf_life_estimates", df)
    # Plot first entity with threshold line
    first_ent = df["entity"].iloc[0] if not df.empty else None
    if first_ent is not None:
        sub = ts.metadata[ts.metadata[ts.entity_col] == first_ent]  # type: ignore[index]
        t = sub[ts.time_col].to_numpy(dtype=float)  # type: ignore[index]
        y = sub[value_col].to_numpy(dtype=float)
        fig, ax = plt.subplots()
        ax.plot(t, y, "o-", label=f"{first_ent}")
        ax.axhline(threshold, color="red", linestyle="--", label="threshold")
        ax.set_xlabel("Time")
        ax.set_ylabel(value_col)
        ax.legend()
        save_figure(report_dir, f"entity_{first_ent}", fig)
        plt.close(fig)
    write_summary_json(
        report_dir,
        {
            "workflow": "shelf_life",
            "n_entities": int(df.shape[0]),
            "value_col": value_col,
            "threshold": float(threshold),
        },
    )
    typer.echo(f"Shelf-life report: {report_dir}")
