from __future__ import annotations
"""
Automated experiment reporting and analysis infrastructure.

Provides protocol-aware HTML report generation, experiment card summarization
with confidence/deployment readiness assessment, and multi-modal risk scoring.

Also includes backward-compatible reporting utilities for CLI commands.
"""

# === NEW REPORTING INFRASTRUCTURE ===
from foodspec.reporting.base import ReportBuilder, ReportContext, collect_figures
from foodspec.reporting.cards import (
    ConfidenceLevel,
    DeploymentReadiness,
    ExperimentCard,
    build_experiment_card,
    build_experiment_card_from_bundle,
)
from foodspec.reporting.compare import compare_runs
from foodspec.reporting.dossier import DossierBuilder, ScientificDossierBuilder
from foodspec.reporting.figures import FigureExporter, FigureStyle, radar_plot
from foodspec.reporting.html import HtmlReportBuilder
from foodspec.reporting.modes import (
    ModeConfig,
    ReportMode,
    get_mode_config,
    list_modes,
    validate_artifacts,
)
from foodspec.reporting.schema import RunBundle

# === BACKWARD COMPATIBILITY: IMPORT FROM DEPRECATED reporting.py ===
# These functions are used by CLI commands and other modules.
# They are defined in the deprecated reporting.py module.
import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None


def create_run_dir(base_dir: pathlib.Path | str, prefix: str) -> pathlib.Path:
    """Create a timestamped run directory under base_dir using UTC."""
    base = pathlib.Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_report_folder(base_out: str | pathlib.Path, workflow_name: str) -> pathlib.Path:
    """Backward-compatible wrapper to create a run directory."""
    return create_run_dir(base_out, workflow_name.upper())


def write_json(path: pathlib.Path | str, data: Dict) -> pathlib.Path:
    """Write JSON with UTF-8 and indentation."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_summary_json(report_dir: pathlib.Path, summary: dict) -> pathlib.Path:
    """Write summary.json to the report directory."""
    report_dir = pathlib.Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    return write_json(report_dir / "summary.json", summary)


def write_metrics_csv(report_dir: pathlib.Path, name: str, df: pd.DataFrame) -> pathlib.Path:
    """Write a metrics DataFrame to CSV with the given base name."""
    report_dir = pathlib.Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{name}.csv"
    df.to_csv(path, index=True)
    return path


def save_figure(report_dir: pathlib.Path, name: str, fig) -> pathlib.Path:
    """Save a matplotlib figure to the report directory as PNG."""
    report_dir = pathlib.Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def summarize_metrics_for_markdown(metrics: Dict[str, Any]) -> str:
    """Convert a nested metrics dict into a Markdown bullet list."""
    lines = []
    for key, val in metrics.items():
        if isinstance(val, dict):
            lines.append(f"- **{key}**:")
            for k2, v2 in val.items():
                lines.append(f"  - {k2}: {v2}")
        else:
            lines.append(f"- **{key}**: {val}")
    return "\n".join(lines)


def write_markdown_report(path: pathlib.Path | str, title: str, sections: Dict[str, str]) -> pathlib.Path:
    """Write a simple Markdown report."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}"]
    for sec_title, body in sections.items():
        parts.append(f"## {sec_title}")
        parts.append(body)
    path.write_text("\n\n".join(parts), encoding="utf-8")
    return path


def summarize_preprocessing_pipeline(pipeline) -> Dict[str, Any]:
    """Summarize a preprocessing pipeline (best-effort stringification)."""
    if pipeline is None:
        return {"pipeline": "None"}
    return {"pipeline": str(pipeline)}


def summarize_stats_results(stats_result) -> Dict[str, Any]:
    """Summarize statistical results into a JSON-friendly dict."""
    if stats_result is None:
        return {}
    if hasattr(stats_result, "summary"):
        return {"summary": stats_result.summary.to_dict(orient="records")}
    if isinstance(stats_result, dict):
        return stats_result
    return {"summary": str(stats_result)}


def summarize_model_performance(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model metrics into a reporting-friendly dict."""
    return metrics or {}


def export_run_metadata(
    output_path: pathlib.Path | str,
    config: Dict[str, Any] | None,
    metrics: Dict[str, Any] | None,
    stats_results: Dict[str, Any] | None,
    extra_info: Dict[str, Any] | None = None,
) -> pathlib.Path:
    """Export run metadata (config, metrics, stats) to JSON for reproducibility."""
    payload: Dict[str, Any] = {
        "config": config or {},
        "metrics": metrics or {},
        "stats": stats_results or {},
    }
    if extra_info:
        payload["extra"] = extra_info
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    return output_path


def export_model_and_metrics(
    model: Any, metrics: Dict[str, Any], path_base: pathlib.Path | str
) -> Dict[str, pathlib.Path]:
    """Save a fitted model and associated metrics side-by-side.

    Parameters
    ----------
    model : Any
        Fitted estimator or pipeline.
    metrics : dict
        Metrics dictionary (classification or regression).
    path_base : Path or str
        Base path without extension; ``.joblib`` and ``.json`` will be written.

    Returns
    -------
    dict
        Paths written: ``{"model": Path, "metrics": Path}``.
    """
    if joblib is None:
        raise ImportError("joblib is required to export models")
    
    base = pathlib.Path(path_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    model_path = base.with_suffix(".joblib")
    metrics_path = base.with_suffix(".json")
    try:
        joblib.dump(model, model_path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not serialize model to {model_path}: {exc}") from exc
    write_json(metrics_path, metrics or {})
    return {"model": model_path, "metrics": metrics_path}


__all__ = [
    # New reporting infrastructure
    "ReportMode",
    "ModeConfig",
    "get_mode_config",
    "list_modes",
    "validate_artifacts",
    "ReportContext",
    "ReportBuilder",
    "collect_figures",
    "ExperimentCard",
    "ConfidenceLevel",
    "DeploymentReadiness",
    "build_experiment_card",
    "build_experiment_card_from_bundle",
    "RunBundle",
    "HtmlReportBuilder",
    "FigureExporter",
    "FigureStyle",
    "radar_plot",
    "compare_runs",
    "DossierBuilder",
    "ScientificDossierBuilder",
    # Backward-compatible utilities
    "create_run_dir",
    "create_report_folder",
    "write_json",
    "write_summary_json",
    "write_metrics_csv",
    "save_figure",
    "summarize_metrics_for_markdown",
    "write_markdown_report",
    "summarize_preprocessing_pipeline",
    "summarize_stats_results",
    "summarize_model_performance",
    "export_run_metadata",
    "export_model_and_metrics",
]
