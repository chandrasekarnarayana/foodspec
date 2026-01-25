from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from foodspec.core.logging import setup_logging
from foodspec.data_objects import ProtocolRunner
from foodspec.io.parsers import read_spectra
from foodspec.io.validators import validate_input
from foodspec.qc.dataset_qc import check_class_balance, diagnose_imbalance
from foodspec.qc.engine import compute_health_scores, detect_outliers
from foodspec.utils.reproducibility import write_reproducibility_snapshot
from foodspec.cli.commands.modeling import qc_command as legacy_qc_command
from foodspec.cli.commands.preprocess import preprocess as legacy_preprocess

io_app = typer.Typer(help="Data extraction and validation.")
qc_app = typer.Typer(help="Quality control commands.")
preprocess_app = typer.Typer(help="Preprocessing commands.")
features_app = typer.Typer(help="Feature extraction commands.")
train_app = typer.Typer(help="Training commands.")


def _init_run_dir(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir=logs_dir)
    write_reproducibility_snapshot(run_dir)
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))

def _update_run_summary(run_dir: Path, payload: dict) -> None:
    summary_path = run_dir / "run_summary.json"
    data = {}
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
        except Exception:
            data = {}
    data.update(payload)
    summary_path.write_text(json.dumps(data, indent=2))


@io_app.command("validate")
def io_validate(
    path: Path = typer.Argument(..., help="Input file or directory."),
    run_dir: Path = typer.Option(Path("runs/io_validate"), help="Run output directory."),
):
    """Validate an input path and inferred format."""

    _init_run_dir(run_dir)
    results = validate_input(str(path))
    _write_json(run_dir / "io_validation.json", results)
    if results["errors"]:
        raise typer.Exit(code=1)
    typer.echo("IO validation complete.")

@qc_app.callback(invoke_without_command=True)
def qc_root(
    ctx: typer.Context,
    input_hdf5: Optional[str] = typer.Argument(None, help="Preprocessed spectra HDF5."),
    model_type: str = typer.Option("oneclass_svm", help="QC model type: oneclass_svm or isolation_forest."),
    label_column: Optional[str] = typer.Option(None, help="Optional label column for inspection."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """QC group (legacy and mindmap commands)."""

    if ctx.invoked_subcommand is None:
        if input_hdf5 is None:
            raise typer.BadParameter("input_hdf5 is required for legacy QC.")
        legacy_qc_command(input_hdf5, model_type, label_column, output_dir)


@qc_app.command("spectral")
def qc_spectral(
    path: Path = typer.Argument(..., help="Spectral file or folder."),
    run_dir: Path = typer.Option(Path("runs/qc_spectral"), help="Run output directory."),
    outlier_method: str = typer.Option("robust_z", help="Outlier detection method."),
):
    """Run spectral QC diagnostics."""

    _init_run_dir(run_dir)
    fs = read_spectra(path)
    health = compute_health_scores(fs)
    outliers = detect_outliers(fs, method=outlier_method)
    payload = {
        "health": health.table.to_dict(orient="list"),
        "health_aggregates": health.aggregates,
        "outliers": {
            "labels": outliers.labels.tolist(),
            "scores": outliers.scores.tolist(),
            "method": outliers.method,
        },
    }
    _write_json(run_dir / "qc_results.json", payload)
    _update_run_summary(run_dir, {"qc": "spectral", "samples": len(outliers.labels)})
    typer.echo("Spectral QC complete.")


@qc_app.command("dataset")
def qc_dataset(
    csv_path: Path = typer.Argument(..., help="CSV with metadata and labels."),
    label_column: str = typer.Option(..., help="Label column."),
    batch_column: Optional[str] = typer.Option(None, help="Optional batch column."),
    run_dir: Path = typer.Option(Path("runs/qc_dataset"), help="Run output directory."),
):
    """Run dataset QC diagnostics."""

    _init_run_dir(run_dir)
    df = pd.read_csv(csv_path)
    balance = check_class_balance(df, label_column)
    dummy = df.copy()
    dummy_x = [[0.0] * 2 for _ in range(len(dummy))]
    from foodspec.core.dataset import FoodSpectrumSet

    ds = FoodSpectrumSet(x=dummy_x, wavenumbers=[1.0, 2.0], metadata=dummy, modality="raman")
    diagnostics = diagnose_imbalance(ds, label_column, stratification_column=batch_column)
    payload = {"balance": balance, "diagnostics": diagnostics}
    _write_json(run_dir / "qc_results.json", payload)
    _update_run_summary(run_dir, {"qc": "dataset", "rows": len(df)})
    typer.echo("Dataset QC complete.")

@preprocess_app.callback(invoke_without_command=True)
def preprocess_root(
    ctx: typer.Context,
    input_folder: Optional[str] = typer.Argument(None, help="Folder containing spectra text files."),
    metadata_csv: Optional[str] = typer.Option(None, help="Optional metadata CSV with sample_id."),
    output_hdf5: Optional[str] = typer.Argument(None, help="Output HDF5 path."),
    modality: str = typer.Option("raman", help="Spectroscopy modality."),
    min_wn: float = typer.Option(600.0, help="Minimum wavenumber for cropping."),
    max_wn: float = typer.Option(1800.0, help="Maximum wavenumber for cropping."),
):
    """Preprocess group (legacy and mindmap commands)."""

    if ctx.invoked_subcommand is None:
        if input_folder is None or output_hdf5 is None:
            raise typer.BadParameter("input_folder and output_hdf5 are required for legacy preprocess.")
        legacy_preprocess(input_folder, metadata_csv, output_hdf5, modality, min_wn, max_wn)


@preprocess_app.command("run")
def preprocess_run(
    protocol_path: Path = typer.Option(..., "--protocol", "-p", help="Protocol YAML/JSON file."),
    input_path: Path = typer.Option(..., "--input", "-i", help="Input CSV or HDF5."),
    run_dir: Path = typer.Option(Path("runs/preprocess"), help="Run output directory."),
):
    """Run preprocessing steps defined in a protocol."""

    _init_run_dir(run_dir)
    runner = ProtocolRunner.from_file(protocol_path)
    runner.config.steps = [s for s in runner.config.steps if s.get("type") == "preprocess"]
    if not runner.config.steps:
        raise typer.BadParameter("Protocol has no preprocess steps.")
    result = runner.run([input_path])
    _update_run_summary(run_dir, {"logs": result.logs, "summary": result.summary})
    typer.echo("Preprocess run complete.")


@features_app.command("extract")
def features_extract(
    protocol_path: Path = typer.Option(..., "--protocol", "-p", help="Protocol YAML/JSON file."),
    input_path: Path = typer.Option(..., "--input", "-i", help="Input CSV or HDF5."),
    run_dir: Path = typer.Option(Path("runs/features"), help="Run output directory."),
):
    """Extract features using a protocol."""

    _init_run_dir(run_dir)
    runner = ProtocolRunner.from_file(protocol_path)
    runner.config.steps = [s for s in runner.config.steps if s.get("type") in {"preprocess", "rq_analysis"}]
    if not runner.config.steps:
        raise typer.BadParameter("Protocol has no preprocess/rq_analysis steps.")
    result = runner.run([input_path])
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables.items():
        try:
            table.to_csv(tables_dir / f"{name}.csv", index=False)
        except Exception:
            pass
    _update_run_summary(run_dir, {"logs": result.logs, "summary": result.summary})
    typer.echo("Feature extraction complete.")


@train_app.command("run")
def train_run(
    protocol_path: Path = typer.Option(..., "--protocol", "-p", help="Protocol YAML/JSON file."),
    input_path: Path = typer.Option(..., "--input", "-i", help="Input CSV or HDF5."),
    run_dir: Path = typer.Option(Path("runs/train"), help="Run output directory."),
):
    """Run a training protocol (full execution)."""

    _init_run_dir(run_dir)
    runner = ProtocolRunner.from_file(protocol_path)
    result = runner.run([input_path])
    _update_run_summary(run_dir, {"logs": result.logs, "summary": result.summary})
    typer.echo("Training run complete.")


@train_app.callback(invoke_without_command=True)
def train_root(
    ctx: typer.Context,
    protocol_path: Optional[Path] = typer.Option(None, "--protocol", "-p", help="Protocol YAML/JSON file."),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Input CSV or HDF5."),
    run_dir: Path = typer.Option(Path("runs/train"), help="Run output directory."),
):
    """Train group (mindmap command alias)."""

    if ctx.invoked_subcommand is None:
        if protocol_path is None or input_path is None:
            raise typer.BadParameter("--protocol and --input are required.")
        train_run(protocol_path, input_path, run_dir)
