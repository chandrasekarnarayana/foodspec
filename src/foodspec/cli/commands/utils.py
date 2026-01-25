"""Utility commands: about, version, report generation."""
from __future__ import annotations


import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import typer

from foodspec import __version__
from foodspec.logging_utils import get_logger, setup_logging
from foodspec.report.methods import MethodsConfig, generate_methods_text
from foodspec.reporting import write_markdown_report
from foodspec.reporting import (
    ReportMode,
    ReportContext,
    ReportBuilder,
    build_experiment_card,
)
from foodspec.reporting.pdf import PDFExporter
from foodspec.utils.reproducibility import write_reproducibility_snapshot

logger = get_logger(__name__)

utils_app = typer.Typer(help="Utility commands")


def _detect_optional_extras() -> list[str]:
    """Detect installed optional extras such as deep learning or gradient boosting."""
    optional_pkgs = ["tensorflow", "torch", "xgboost", "lightgbm"]
    installed = []
    for pkg in optional_pkgs:
        if importlib.util.find_spec(pkg) is not None:
            installed.append(pkg)
    return installed


@utils_app.command("about")
def about() -> None:
    """Print version and environment information for foodspec."""
    extras = _detect_optional_extras()
    typer.echo(f"foodspec version: {__version__}")
    typer.echo(f"Python version: {sys.version.split()[0]}")
    typer.echo(f"Optional extras detected: {', '.join(extras) if extras else 'none'}")
    typer.echo("Documentation: https://github.com/your-org/foodspec#documentation")
    typer.echo("Description: foodspec is a headless, research-grade toolkit for Raman/FTIR in food science.")


@utils_app.command("report")
def report(
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
    """Generate a report from a run directory or a methods.md document."""
    if run_dir is not None:
        run_path = Path(run_dir).resolve()
        if not run_path.exists():
            typer.echo(f"❌ Run directory not found: {run_path}", err=True)
            raise typer.Exit(1)
        logs_dir = run_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(run_dir=logs_dir)
        write_reproducibility_snapshot(run_path)
        html_path = run_path / "report.html"
        html_path.write_text(
            f"<html><body><h1>{title}</h1>"
            f"<p>Run directory: {run_path}</p>"
            "<p>See run_summary.json and tables for details.</p>"
            "</body></html>"
        )
        if format == "pdf":
            exporter = PDFExporter()
            pdf_path = run_path / "report.pdf"
            exporter.export(html_path, pdf_path)
        summary_path = run_path / "run_summary.json"
        payload = {"report_format": format, "run_dir": str(run_path)}
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text())
            except Exception:
                data = {}
            data.update(payload)
        else:
            data = payload
        summary_path.write_text(json.dumps(data, indent=2))
        typer.echo("Report generated.")
        return

    if dataset is None or sample_size is None or target is None:
        raise typer.BadParameter("dataset, sample_size, and target are required for methods report.")
    cfg = MethodsConfig(
        dataset=dataset,
        sample_size=sample_size,
        target=target,
        modality=modality,
        instruments=[s.strip() for s in (instruments or "").split(",") if s.strip()],
        preprocessing=[s.strip() for s in (preprocessing or "").split(",") if s.strip()],
        models=[s.strip() for s in (models or "").split(",") if s.strip()],
        metrics=[s.strip() for s in (metrics or "").split(",") if s.strip()],
    )
    text = generate_methods_text(cfg, style=style)  # type: ignore[arg-type]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_markdown_report(out_path / "methods.md", title="Methods", sections={"Methods": text})
    print(f"Wrote methods.md to {out_path}")

@utils_app.command("report-run")
def report_run(
    run_dir: Path = typer.Option(
        ..., "--run-dir", "-r", help="Path to protocol run directory."
    ),
    mode: str = typer.Option(
        "research",
        "--mode", "-m",
        help="Report mode: research|regulatory|monitoring",
    ),
    output_format: str = typer.Option(
        "all",
        "--format", "-f",
        help="Output format: html|json|markdown|all",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Output directory (defaults to run_dir)",
    ),
    title: str = typer.Option(
        "FoodSpec Report", "--title", "-t",
        help="Report title for HTML output",
    ),
):
    """Generate automated experiment report from protocol run artifacts.

    Creates HTML report, experiment card (JSON/Markdown), and risk assessment
    from a protocol run directory.

    Example:
        foodspec report-run --run-dir ./protocol_runs/20260125_123456_run/
        foodspec report-run --run-dir ./run/ --mode regulatory --format all
    """
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        typer.echo(f"❌ Run directory not found: {run_path}", err=True)
        raise typer.Exit(1)

    output_path = Path(out_dir or run_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Validate mode
        mode_lower = mode.lower()
        if mode_lower == "research":
            report_mode = ReportMode.RESEARCH
        elif mode_lower == "regulatory":
            report_mode = ReportMode.REGULATORY
        elif mode_lower == "monitoring":
            report_mode = ReportMode.MONITORING
        else:
            typer.echo(
                f"❌ Invalid mode '{mode}'. Choose: research|regulatory|monitoring",
                err=True,
            )
            raise typer.Exit(1)

        # Load context
        typer.echo(f"📂 Loading artifacts from {run_path}...")
        context = ReportContext.load(run_path)
        typer.echo(f"✓ Loaded: {', '.join(context.available_artifacts())}")

        # Generate outputs
        output_format_lower = output_format.lower()
        formats = []
        if output_format_lower in ("all", "html"):
            formats.append("html")
        if output_format_lower in ("all", "json"):
            formats.append("json")
        if output_format_lower in ("all", "markdown"):
            formats.append("markdown")

        if not formats:
            typer.echo(
                f"❌ Invalid format '{output_format}'. Choose: html|json|markdown|all",
                err=True,
            )
            raise typer.Exit(1)

        # HTML Report
        if "html" in formats:
            typer.echo("📄 Generating HTML report...")
            html_path = output_path / "report.html"
            ReportBuilder(context).build_html(html_path, mode=report_mode, title=title)
            typer.echo(f"✓ HTML report: {html_path}")

        # Experiment Card (JSON + Markdown)
        typer.echo("🎯 Building experiment card...")
        card = build_experiment_card(context, mode=report_mode)

        if "json" in formats:
            json_path = output_path / "card.json"
            card.to_json(json_path)
            typer.echo(f"✓ JSON card: {json_path}")

        if "markdown" in formats:
            md_path = output_path / "card.md"
            card.to_markdown(md_path)
            typer.echo(f"✓ Markdown card: {md_path}")

        # Summary
        typer.echo("\n✅ Report generation complete!")
        typer.echo(f"\nCard Summary:")
        typer.echo(f"  Confidence: {card.confidence_level.value}")
        typer.echo(f"  Deployment: {card.deployment_readiness.value}")
        if card.key_risks:
            typer.echo(f"  Risks: {len(card.key_risks)}")
            for risk in card.key_risks[:3]:
                typer.echo(f"    - {risk}")
            if len(card.key_risks) > 3:
                typer.echo(f"    ... and {len(card.key_risks) - 3} more")
        else:
            typer.echo(f"  Risks: None")

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        logger.exception("Report generation failed")
        raise typer.Exit(1)
