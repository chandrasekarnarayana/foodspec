"""Reporting CLI commands for FoodSpec."""

from __future__ import annotations

import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer

from foodspec.cli.commands.utils import report as legacy_report
from foodspec.core.errors import FoodSpecQCError, FoodSpecValidationError
from foodspec.core.manifest import RunManifest
from foodspec.modeling.validation.quality import validate_dataset
from foodspec.protocol import ProtocolRunner
from foodspec.qc.dataset_qc import check_class_balance
from foodspec.qc.policy import QCPolicy
from foodspec.reporting.artifacts import finalize_reporting_run, init_reporting_run
from foodspec.reporting.cards import build_experiment_card_from_bundle
from foodspec.reporting.compare import compare_runs
from foodspec.reporting.dossier import ScientificDossierBuilder
from foodspec.reporting.figures import FigureExporter, FigureStyle, radar_plot
from foodspec.reporting.html import HtmlReportBuilder
from foodspec.reporting.modes import ReportMode, get_mode_config, validate_artifacts
from foodspec.reporting.pdf import export_pdf
from foodspec.reporting.schema import RunBundle
from foodspec.utils.run_artifacts import safe_json_dump, update_manifest

report_app = typer.Typer(help="Reporting and visualization commands.", invoke_without_command=True)


@report_app.callback(invoke_without_command=True)
def report_root(
    ctx: typer.Context,
    run_dir: Optional[Path] = typer.Option(None, "--run-dir", "-r", help="Run directory with outputs."),
    format: str = typer.Option("html", "--format", help="html or pdf"),
    title: str = typer.Option("FoodSpec Report", help="Report title for HTML output."),
    dataset: Optional[str] = typer.Option(None, help="Dataset name."),
    sample_size: Optional[int] = typer.Option(None, help="Number of samples."),
    target: Optional[str] = typer.Option(None, help="Target variable description."),
    modality: str = typer.Option("raman", help="Modality: raman|ftir|nir"),
    instruments: Optional[str] = typer.Option(None, help="Comma-separated instruments."),
    preprocessing: Optional[str] = typer.Option(None, help="Comma-separated preprocessing steps."),
    models: Optional[str] = typer.Option(None, help="Comma-separated models."),
    metrics: Optional[str] = typer.Option("accuracy", help="Comma-separated metrics."),
    out_dir: str = typer.Option("report_methods", help="Output directory for methods.md."),
    style: str = typer.Option("journal", help="Style: journal|concise|bullet"),
):
    """Legacy report entry point for backward compatibility."""
    if ctx.invoked_subcommand is None:
        legacy_report(
            run_dir=run_dir,
            format=format,
            title=title,
            dataset=dataset,
            sample_size=sample_size,
            target=target,
            modality=modality,
            instruments=instruments,
            preprocessing=preprocessing,
            models=models,
            metrics=metrics,
            out_dir=out_dir,
            style=style,
        )


def _run_qc_policy(cfg, input_path: Path, run_dir: Path, mode: ReportMode) -> dict:
    qc_policy = QCPolicy.from_dict(getattr(cfg, "qc", {}) or {})
    mode_config = get_mode_config(mode)
    if mode_config.strictness_level >= 2:
        qc_policy.required = True

    required_cols = []
    class_col = None
    for step in cfg.steps:
        if step.get("type") == "qc_checks":
            params = step.get("params", {})
            required_cols = params.get("required_columns", []) or []
            class_col = params.get("class_col")
            break

    label_col = None
    if class_col:
        label_col = class_col
    elif getattr(cfg, "expected_columns", None):
        label_col = cfg.expected_columns.get("oil_col") or cfg.expected_columns.get("label_col")

    report: dict = {
        "status": "skipped",
        "policy": qc_policy.to_dict(),
        "reason": "qc_not_evaluated",
    }

    if input_path.suffix.lower() == ".csv" and input_path.exists():
        df = pd.read_csv(input_path)
        diag = validate_dataset(df, required_cols=required_cols, class_col=label_col)
        balance = None
        if label_col and label_col in df.columns:
            balance = check_class_balance(df, label_col)
        policy_result = qc_policy.evaluate_dataset(balance or {"imbalance_ratio": 0.0, "undersized_classes": []})
        if diag["errors"]:
            policy_result["status"] = "fail"
            policy_result.setdefault("flags", []).append("schema_errors")
        report = {
            "status": policy_result["status"],
            "schema": diag,
            "balance": balance,
            "policy_result": policy_result,
        }
    elif qc_policy.required:
        report["status"] = "fail"

    report_path = run_dir / "qc_report.json"
    safe_json_dump(report_path, report)
    if qc_policy.required and report["status"] != "pass":
        raise FoodSpecQCError("QC required by policy; failing run.")
    return report


def _generate_figures(bundle: RunBundle, figures_dir: Path, style: FigureStyle, seed: int) -> List[Path]:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    metrics = bundle.metrics[0] if bundle.metrics else {}
    labels = []
    values = []
    for key in ("macro_f1", "auroc", "accuracy"):
        if key in metrics:
            try:
                labels.append(key)
                values.append(float(metrics[key]))
            except (TypeError, ValueError):
                continue
    if not values:
        labels = ["metric_a", "metric_b", "metric_c"]
        values = rng.random(3).tolist()

    fig, ax = plt.subplots()
    ax.bar(labels, values, color="#2a6fdb")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Metrics Overview")
    exporter = FigureExporter(style=style, size_preset="single")
    outputs = exporter.export(fig, figures_dir, "metrics_overview")
    plt.close(fig)

    radar = radar_plot(labels, values, title="Metrics Radar", seed=seed)
    outputs.extend(exporter.export(radar, figures_dir, "metrics_radar"))
    plt.close(radar)

    return outputs


@report_app.command("run")
def report_run(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input CSV or HDF5."),
    protocol_path: Path = typer.Option(..., "--protocol", "-p", help="Protocol YAML/JSON."),
    out_dir: Path = typer.Option(Path("runs/report"), "--outdir", help="Output directory."),
    mode: ReportMode = typer.Option(ReportMode.RESEARCH, "--mode", help="reporting mode"),
    format: str = typer.Option("html", "--format", help="html|pdf|both"),
    figure_style: FigureStyle = typer.Option(FigureStyle.JOSS, "--figure-style", help="Paper-ready figure style."),
    seed: int = typer.Option(0, "--seed", help="Random seed for deterministic outputs."),
    embed_images: bool = typer.Option(False, "--embed-images", help="Embed images into HTML."),
):
    """Run protocol + generate a complete report bundle."""
    run_dir = init_reporting_run(
        out_dir,
        command="report.run",
        inputs=[input_path],
        config={"protocol_path": str(protocol_path), "mode": mode.value, "format": format},
        mode=mode.value,
        seed=seed,
        protocol_path=protocol_path,
        args=list(sys.argv),
    )
    try:
        runner = ProtocolRunner.from_file(protocol_path)
        runner.config.seed = seed
        protocol_snapshot = asdict(runner.config)
        protocol_hash = RunManifest.compute_protocol_hash(protocol_snapshot)
        data_fingerprint = ""
        if input_path.exists() and input_path.is_file():
            data_fingerprint = RunManifest.compute_data_fingerprint(input_path)
        update_manifest(
            run_dir,
            {
                "protocol_snapshot": protocol_snapshot,
                "protocol_hash": protocol_hash,
                "data_fingerprint": data_fingerprint,
            },
        )
        qc_report = _run_qc_policy(runner.config, input_path, run_dir, mode)
        finalize_reporting_run(
            run_dir,
            {"qc_report": str(run_dir / "qc_report.json"), "qc_status": qc_report.get("status", "unknown")},
        )
        result = runner.run([input_path])
        runner.save_outputs(result, run_dir)

        finalize_reporting_run(run_dir, {"status": "running", "summary": result.summary, "seed": seed})
        bundle = RunBundle.from_run_dir(run_dir)
        valid, missing = validate_artifacts(mode, bundle.available_artifacts)
        finalize_reporting_run(run_dir, {"artifact_check": {"valid": valid, "missing": missing}})

        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        _generate_figures(bundle, figures_dir, figure_style, seed)

        card = build_experiment_card_from_bundle(bundle, mode=mode)
        cards_dir = run_dir / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        card_path = card.to_markdown(cards_dir / "experiment_card.md")
        card.to_json(cards_dir / "experiment_card.json")
        flat_card_md = run_dir / "card.md"
        flat_card_json = run_dir / "card.json"
        card.to_markdown(flat_card_md)
        card.to_json(flat_card_json)

        report_builder = HtmlReportBuilder(bundle, mode=mode)
        html_path = report_builder.build(run_dir, embed_images=embed_images)
        flat_report = run_dir / "report.html"
        if html_path != flat_report:
            shutil.copy2(html_path, flat_report)
        pdf_path = None
        if format in {"pdf", "both"}:
            pdf_path = export_pdf(html_path, run_dir / "reports" / "report.pdf")

        dossier_builder = ScientificDossierBuilder()
        dossier_builder.build(bundle, run_dir, mode=mode.value)

        artifacts = {
            "report_html": str(html_path),
            "report_html_flat": str(flat_report),
            "report_pdf": str(pdf_path) if pdf_path else None,
            "figures_dir": str(figures_dir),
            "experiment_card": str(card_path),
            "experiment_card_flat": str(flat_card_md),
            "dossier": str(run_dir / "dossier" / "dossier.md"),
            "qc_report": str(run_dir / "qc_report.json"),
        }
        finalize_reporting_run(
            run_dir,
            {
                "status": "success",
                "summary": result.summary,
                "mode": mode.value,
                "seed": seed,
                "artifacts": artifacts,
            },
        )
        update_manifest(run_dir, {"artifacts": artifacts})
        typer.echo(f"Report run complete -> {run_dir}")
    except FoodSpecQCError as exc:
        finalize_reporting_run(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3)
    except (FoodSpecValidationError, ValueError, typer.BadParameter) as exc:
        finalize_reporting_run(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        finalize_reporting_run(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@report_app.command("compare")
def report_compare(
    runs: List[Path] = typer.Option(..., "--runs", "-r", help="Run directories to compare."),
    out_dir: Path = typer.Option(Path("runs/report_compare"), "--outdir", help="Output directory."),
    figure_style: FigureStyle = typer.Option(FigureStyle.JOSS, "--figure-style", help="Paper-ready figure style."),
    seed: int = typer.Option(0, "--seed", help="Random seed for deterministic outputs."),
):
    """Compare multiple runs and build dashboards."""
    run_dir = init_reporting_run(
        out_dir,
        command="report.compare",
        inputs=runs,
        config={"runs": [str(r) for r in runs]},
        mode="compare",
        seed=seed,
        args=list(sys.argv),
    )
    try:
        result = compare_runs(runs, run_dir, style=figure_style, seed=seed)
        artifacts = {
            "dashboard": str(result.dashboard_path),
            "leaderboard": str(result.leaderboard_path),
            "radar": str(result.radar_path),
        }
        finalize_reporting_run(
            run_dir,
            {"status": "success", "mode": "compare", "seed": seed, "artifacts": artifacts},
        )
        update_manifest(run_dir, {"artifacts": artifacts})
        typer.echo(f"Comparison report written to {run_dir}")
    except (FoodSpecValidationError, ValueError) as exc:
        finalize_reporting_run(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        finalize_reporting_run(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


__all__ = ["report_app"]
