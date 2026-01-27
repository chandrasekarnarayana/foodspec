from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from foodspec.workflow.orchestrator import EndToEndOrchestrator

run_e2e_app = typer.Typer(
    help="End-to-end orchestrated run: validate -> preprocess -> features -> model -> trust -> report.",
    invoke_without_command=True,
)


@run_e2e_app.callback()
def run_e2e(
    csv: Path = typer.Option(..., "--csv", help="Input CSV with spectra and metadata."),
    protocol: Path = typer.Option(..., "--protocol", "-p", help="Protocol YAML/JSON file."),
    scheme: Optional[str] = typer.Option(None, "--scheme", help="Validation scheme (loso|lobo|nested|random)."),
    group: Optional[str] = typer.Option(None, "--group", help="Group column for LOSO/LOBO."),
    model: str = typer.Option("lightgbm", "--model", help="Model backend (e.g., lightgbm, logreg, rf)."),
    mode: str = typer.Option("research", "--mode", help="Report mode: research|regulatory|monitoring."),
    features: str = typer.Option("peaks", "--features", help="Feature backend: raw|peaks|bands|pca|pls|hybrid."),
    outdir: Path = typer.Option(Path("runs/run_e2e"), "--outdir", help="Output directory."),
    label_col: Optional[str] = typer.Option(None, "--label-col", help="Label column override."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    trust: bool = typer.Option(True, "--trust/--no-trust", help="Run trust stack (calibration/conformal/abstention)."),
    viz: bool = typer.Option(True, "--viz/--no-viz", help="Generate basic visualizations."),
    report: bool = typer.Option(True, "--report/--no-report", help="Generate HTML report."),
    pdf: bool = typer.Option(False, "--pdf", help="Also export PDF if possible."),
    unsafe_random_cv: bool = typer.Option(False, "--unsafe-random-cv", help="Allow random CV when no group column is available."),
):
    """Run a complete FoodSpec pipeline (schema->preprocess->features->model->trust->report)."""
    orchestrator = EndToEndOrchestrator(
        csv_path=csv,
        protocol_path=protocol,
        out_dir=outdir,
        scheme=scheme or "",
        model=model,
        feature_type=features,
        label_col=label_col,
        group_col=group,
        mode=mode,
        enable_trust=trust,
        enable_viz=viz,
        enable_report=report,
        generate_pdf=pdf,
        seed=seed,
        unsafe_random_cv=unsafe_random_cv,
    )
    result = orchestrator.run()
    if result.status == "success":
        raise typer.Exit(code=0)
    if result.status == "validation_error":
        raise typer.Exit(code=2)
    if result.status == "qc_fail":
        raise typer.Exit(code=3)
    raise typer.Exit(code=4)


__all__ = ["run_e2e_app"]
