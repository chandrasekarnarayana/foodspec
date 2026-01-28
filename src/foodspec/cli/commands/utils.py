"""Utility commands: about, version, report generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import typer

from foodspec import __version__
from foodspec.logging_utils import get_logger
from foodspec.report.methods import MethodsConfig, generate_methods_text
from foodspec.reporting import write_markdown_report
from foodspec.reporting.api import build_report_from_run
from foodspec.reporting.pdf import PDFExporter
from foodspec.utils.run_artifacts import (
    get_logger as get_run_logger,
)
from foodspec.utils.run_artifacts import (
    init_run_dir,
    write_manifest,
    write_run_summary,
)

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
    metrics: Optional[str] = typer.Option("accuracy", help="Comma-separated metrics."),
    dataset: Optional[str] = typer.Option(None, help="Dataset name for methods report."),
    sample_size: Optional[int] = typer.Option(None, help="Sample size for methods report."),
    target: Optional[str] = typer.Option(None, help="Target variable for methods report."),
    modality: str = typer.Option("raman", help="Modality for methods report."),
    instruments: Optional[str] = typer.Option(None, help="Comma-separated instruments."),
    preprocessing: Optional[str] = typer.Option(None, help="Comma-separated preprocessing steps."),
    models: Optional[str] = typer.Option(None, help="Comma-separated model names."),
    out_dir: str = typer.Option("report_methods", help="Output directory for methods.md."),
    style: str = typer.Option("journal", help="Style: journal|concise|bullet"),
):
    """Generate a report from a run directory or a methods.md document."""
    if run_dir is not None:
        run_path = Path(run_dir).resolve()
        exists = run_path.exists()
        run_path = init_run_dir(run_path)
        get_run_logger(run_path)
        write_manifest(run_path, {"command": "report", "inputs": [run_path], "format": format})
        if not exists:
            msg = f"Run directory not found: {run_path}"
            write_run_summary(run_path, {"status": "fail", "error": msg})
            typer.echo(f"ERROR: {msg}", err=True)
            raise typer.Exit(2)
        try:
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
            write_run_summary(run_path, {"status": "success", "report_format": format, "run_dir": str(run_path)})
            write_manifest(
                run_path,
                {
                    "command": "report",
                    "inputs": [run_path],
                    "format": format,
                    "artifacts": {"report": str(html_path), "run_summary": "run_summary.json"},
                },
            )
            typer.echo("Report generated.")
            return
        except Exception as exc:
            write_run_summary(run_path, {"status": "fail", "error": str(exc)})
            typer.echo(f"Runtime error: {exc}", err=True)
            raise typer.Exit(code=4)

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
    out_path = init_run_dir(Path(out_dir))
    get_run_logger(out_path)
    write_manifest(out_path, {"command": "report", "inputs": [], "mode": "methods"})
    try:
        write_markdown_report(out_path / "methods.md", title="Methods", sections={"Methods": text})
        write_run_summary(out_path, {"status": "success", "methods_path": str(out_path / "methods.md")})
        write_manifest(
            out_path,
            {
                "command": "report",
                "inputs": [],
                "mode": "methods",
                "artifacts": {"methods": str(out_path / "methods.md"), "run_summary": "run_summary.json"},
            },
        )
        print(f"Wrote methods.md to {out_path}")
    except Exception as exc:
        write_run_summary(out_path, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@utils_app.command("report-run")
def report_run(
    run_dir: Path = typer.Option(..., "--run-dir", "-r", help="Path to protocol run directory."),
    mode: str = typer.Option(
        "research",
        "--mode",
        "-m",
        help="Report mode: research|regulatory|monitoring",
    ),
    output_format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Output format: html|json|markdown|all",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (defaults to run_dir)",
    ),
    title: str = typer.Option(
        "FoodSpec Report",
        "--title",
        "-t",
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
    output_path = Path(out_dir or run_dir).resolve()
    run_dir_path = init_run_dir(output_path)
    get_run_logger(run_dir_path)
    write_manifest(run_dir_path, {"command": "report-run", "inputs": [run_path]})
    if not run_path.exists():
        msg = f"Run directory not found: {run_path}"
        write_run_summary(run_dir_path, {"status": "fail", "error": msg})
        typer.echo(f"ERROR: {msg}", err=True)
        raise typer.Exit(2)
    try:
        mode_lower = mode.lower()
        if mode_lower not in {"research", "regulatory", "monitoring"}:
            msg = f"Invalid mode '{mode}'. Choose: research|regulatory|monitoring"
            write_run_summary(run_dir_path, {"status": "fail", "error": msg})
            typer.echo(f"ERROR: {msg}", err=True)
            raise typer.Exit(2)

        output_format_lower = output_format.lower()
        formats = []
        if output_format_lower in ("all", "html"):
            formats.append("html")
        if output_format_lower in ("all", "json"):
            formats.append("json")
        if output_format_lower in ("all", "markdown"):
            formats.append("markdown")
        want_pdf = output_format_lower in {"all", "pdf"}

        if not formats and not want_pdf:
            msg = f"Invalid format '{output_format}'. Choose: html|json|markdown|all"
            write_run_summary(run_dir_path, {"status": "fail", "error": msg})
            typer.echo(f"ERROR: {msg}", err=True)
            raise typer.Exit(2)

        artifacts = build_report_from_run(
            run_path,
            out_dir=run_dir_path,
            mode=mode_lower,
            pdf=want_pdf,
            title=title,
        )
        artifacts["run_summary"] = "run_summary.json"

        write_run_summary(
            run_dir_path,
            {
                "status": "success",
                "report_mode": mode_lower,
                "report_formats": formats,
                "artifacts": artifacts,
            },
        )
        write_manifest(
            run_dir_path,
            {
                "command": "report-run",
                "inputs": [run_path],
                "mode": mode_lower,
                "format": output_format_lower,
                "artifacts": artifacts,
            },
        )

        # Summary
        typer.echo("Report generation complete.")
        typer.echo(f"Artifacts: {', '.join(sorted(artifacts.keys()))}")

    except Exception as e:
        write_run_summary(run_dir_path, {"status": "fail", "error": str(e)})
        typer.echo(f"ERROR: {str(e)}", err=True)
        logger.exception("Report generation failed")
        raise typer.Exit(4)
