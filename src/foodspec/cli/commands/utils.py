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
    dataset: str = typer.Option(..., help="Dataset name."),
    sample_size: int = typer.Option(..., help="Number of samples."),
    target: str = typer.Option(..., help="Target variable description."),
    modality: str = typer.Option("raman", help="Modality: raman|ftir|nir"),
    instruments: Optional[str] = typer.Option(None, help="Comma-separated instruments."),
    preprocessing: Optional[str] = typer.Option(None, help="Comma-separated preprocessing steps."),
    models: Optional[str] = typer.Option(None, help="Comma-separated models."),
    metrics: Optional[str] = typer.Option("accuracy", help="Comma-separated metrics."),
    out_dir: str = typer.Option("report_methods", help="Output directory for methods.md."),
    style: str = typer.Option("journal", help="Style: journal|concise|bullet"),
):
    """Generate a paper-ready methods.md from structured inputs."""
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
