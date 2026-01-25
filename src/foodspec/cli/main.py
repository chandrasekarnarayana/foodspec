from __future__ import annotations
"""Command-line interface for foodspec.

This module assembles CLI commands organized into logical groups:
- Data management (csv-to-library, library-search, library-auth, model-info)
- Preprocessing (preprocess)
- Modeling (qc, fit, predict)
- Analysis (oil-auth, heating, domains, mixture, hyperspectral, aging, shelf-life)
- Workflow orchestration (run-exp, protocol-benchmarks, bench)
- Utilities (about, report)
"""


from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Optional, List, Union

import matplotlib
import typer

matplotlib.use("Agg")

from foodspec.cli.commands.analysis import analysis_app
from foodspec.cli.commands.data import data_app
from foodspec.cli.commands.mindmap import (
    features_app as mindmap_features_app,
    io_app as mindmap_io_app,
    preprocess_app as mindmap_preprocess_app,
    qc_app as mindmap_qc_app,
    train_app as mindmap_train_app,
)
from foodspec.cli.commands.modeling import modeling_app
from foodspec.cli.commands.utils import utils_app
from foodspec.cli.commands.workflow import workflow_app
from foodspec.core.spectral_dataset import HyperspectralDataset, SpectralDataset
from foodspec.protocol import ProtocolRunner, load_protocol, validate_protocol
import pandas as pd
import glob
import json

app = typer.Typer(help="foodspec command-line interface")

# Mindmap-aligned commands (new structure)
app.add_typer(mindmap_io_app, name="io")
app.add_typer(mindmap_qc_app, name="qc")
app.add_typer(mindmap_preprocess_app, name="preprocess")
app.add_typer(mindmap_features_app, name="features")
app.add_typer(mindmap_train_app, name="train")


# Global options for root command
@app.callback(invoke_without_command=True)
def root_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show FoodSpec version and exit",
    ),
):
    """Root-level options for the foodspec CLI."""
    if version:
        try:
            v = pkg_version("foodspec")
        except PackageNotFoundError:
            v = "unknown"
        typer.echo(v)
        raise typer.Exit()


# Register command groups
# Data commands
app.command("csv-to-library")(data_app.registered_commands[0].callback)
app.command("library-search")(data_app.registered_commands[1].callback)
app.command("library-auth")(data_app.registered_commands[2].callback)
app.command("model-info")(data_app.registered_commands[3].callback)

# Modeling
app.command("fit")(modeling_app.registered_commands[1].callback)
app.command("predict")(modeling_app.registered_commands[2].callback)

# Analysis
app.command("oil-auth")(analysis_app.registered_commands[0].callback)
app.command("heating")(analysis_app.registered_commands[1].callback)
app.command("domains")(analysis_app.registered_commands[2].callback)
app.command("mixture")(analysis_app.registered_commands[3].callback)
app.command("hyperspectral")(analysis_app.registered_commands[4].callback)
app.command("aging")(analysis_app.registered_commands[5].callback)
app.command("shelf-life")(analysis_app.registered_commands[6].callback)

# Workflow
app.command("run-exp")(workflow_app.registered_commands[0].callback)
app.command("protocol-benchmarks")(workflow_app.registered_commands[1].callback)
app.command("bench")(workflow_app.registered_commands[2].callback)

# Utilities
app.command("about")(utils_app.registered_commands[0].callback)
app.command("report")(utils_app.registered_commands[1].callback)
app.command("report-run")(utils_app.registered_commands[2].callback)


# --- Protocol runner (convenience) ------------------------------------------------------

@app.command("run")
def run_protocol(
    protocol: str = typer.Option(..., "--protocol", "-p", help="Protocol name or path to YAML/JSON."),
    input: List[Path] = typer.Option(None, "--input", "-i", help="Input CSV/HDF5 file (repeatable)."),
    input_dir: Optional[Path] = typer.Option(None, help="Directory of inputs when using --glob."),
    glob_pattern: str = typer.Option("*.csv", "--glob", help="Glob pattern used with --input-dir."),
    output_dir: Path = typer.Option(Path("protocol_runs"), "--output-dir", "--outdir", help="Directory for run outputs."),
    seed: Optional[int] = typer.Option(None, help="Random seed override."),
    cv_folds: Optional[int] = typer.Option(None, help="Override CV folds for RQ models."),
    normalization_mode: Optional[str] = typer.Option(None, help="Normalization mode override (e.g. reference)."),
    baseline_method: Optional[str] = typer.Option(None, help="Baseline method override (als, rubberband, none)."),
    spike_removal: Optional[bool] = typer.Option(
        None,
        "--spike-removal/--no-spike-removal",
        help="Enable/disable spike removal in preprocessing.",
    ),
    validation_strategy: Optional[str] = typer.Option(
        None,
        help="Validation strategy (standard, batch_aware, group_stratified).",
    ),
    report_level: str = typer.Option("standard", help="Report richness for auto-publish."),
    auto: bool = typer.Option(False, help="Generate a markdown bundle after run."),
    dry_run: bool = typer.Option(False, help="Validate only; do not execute."),
    verbose: bool = typer.Option(False, help="Verbose logging."),
    quiet: bool = typer.Option(False, help="Quiet mode (less output)."),
):
    """Run a FoodSpec protocol on one or more inputs.

    Example: foodspec run --protocol examples/protocols/EdibleOil_Classification_v1.yaml \
             --input data/oils.csv --outdir runs/exp1 --seed 0
    """
    # Collect inputs
    inputs: List[Path] = []
    if input:
        inputs.extend([Path(p) for p in input])
    if input_dir:
        inputs.extend([Path(p) for p in glob.glob(str(Path(input_dir) / glob_pattern))])
    if not inputs:
        raise typer.BadParameter("No inputs provided. Use --input or --input-dir.")

    # Load protocol
    proto_path = Path(protocol)
    runner: ProtocolRunner
    if proto_path.exists():
        runner = ProtocolRunner.from_file(proto_path)
    else:
        runner = ProtocolRunner(load_protocol(protocol))

    cfg = runner.config
    if seed is not None:
        cfg.seed = seed
    if validation_strategy:
        cfg.validation_strategy = validation_strategy
    # Apply overrides to steps
    for step in cfg.steps:
        if step.get("type") == "preprocess":
            if normalization_mode:
                step.setdefault("params", {})["normalization"] = normalization_mode
            if baseline_method:
                step.setdefault("params", {})["baseline_method"] = baseline_method
            if spike_removal is not None:
                step.setdefault("params", {})["spike_removal"] = spike_removal
        if step.get("type") == "rq_analysis":
            if cv_folds:
                step.setdefault("params", {})["n_splits"] = cv_folds
            if normalization_mode:
                step.setdefault("params", {})["normalization_modes"] = [normalization_mode]

    # Print quick summary
    if not quiet:
        typer.echo(f"=== FoodSpec Protocol Runner ===")
        typer.echo(f"Protocol: {cfg.name} (v{cfg.version})")
        typer.echo(f"Inputs: {len(inputs)} file(s)")
    if dry_run:
        # Perform validation on the first CSV input if possible
        try:
            first = inputs[0]
            df_or_obj: Union[pd.DataFrame, SpectralDataset, HyperspectralDataset]
            if first.suffix.lower() in {".h5", ".hdf5"}:
                try:
                    df_or_obj = SpectralDataset.from_hdf5(first)
                except Exception:
                    df_or_obj = HyperspectralDataset.from_hdf5(first)
            else:
                df_or_obj = pd.read_csv(first)
            if isinstance(df_or_obj, pd.DataFrame):
                diag = validate_protocol(cfg, df_or_obj)
                if diag["errors"]:
                    typer.echo("Validation errors:")
                    for e in diag["errors"]:
                        typer.echo(f" - {e}")
                    raise typer.Exit(code=1)
                if diag["warnings"] and not quiet:
                    typer.echo("Validation warnings:")
                    for w in diag["warnings"]:
                        typer.echo(f" - {w}")
        except Exception:
            pass
        typer.echo("Dry-run complete (no execution).")
        raise typer.Exit(code=0)

    # Execute per input and save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        if verbose and not quiet:
            typer.echo(f"[INFO] Running protocol on {path}")
        if path.suffix.lower() in {".h5", ".hdf5"}:
            try:
                df_or_obj2 = SpectralDataset.from_hdf5(path)
            except Exception:
                df_or_obj2 = HyperspectralDataset.from_hdf5(path)
        else:
            df_or_obj2 = pd.read_csv(path)

        # Validate when DataFrame
        if isinstance(df_or_obj2, pd.DataFrame):
            diag2 = validate_protocol(cfg, df_or_obj2)
            if diag2["errors"]:
                for e in diag2["errors"]:
                    if not quiet:
                        typer.echo(f"ERROR: {e}")
                continue

        # Run with basic safety on n_splits if needed
        try:
            result = runner.run([df_or_obj2])
        except Exception as exc:
            if "n_splits" in str(exc):
                for step in cfg.steps:
                    if step.get("type") == "rq_analysis":
                        step.setdefault("params", {})["n_splits"] = max(2, cv_folds or 2)
                result = runner.run([df_or_obj2])
            else:
                raise

        target = output_dir / f"{cfg.name}_{path.stem}"
        runner.save_outputs(result, target)

        # Annotate metadata with inputs list
        meta_path = target / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta["inputs"] = [str(p) for p in inputs]
                meta["multi_input"] = len(inputs) > 1
                meta_path.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

        if not quiet:
            report_html = target / "report.html"
            typer.echo(f"Run complete → {target}")
            if report_html.exists():
                typer.echo(f"Report: {report_html}")
        # Optional auto-publish (markdown bundle)
        if auto:
            try:
                from foodspec.narrative import save_markdown_bundle

                publish_dir = target / "publish"
                publish_dir.mkdir(parents=True, exist_ok=True)
                fig_limit = {"summary": 4, "standard": 8, "full": None}.get(report_level, 8)
                save_markdown_bundle(
                    target,
                    publish_dir,
                    fig_limit=fig_limit,
                    include_all=(report_level == "full"),
                    profile="standard" if report_level != "summary" else "quicklook",
                )
                if not quiet:
                    typer.echo(f"Auto-publish bundle saved → {publish_dir}")
            except Exception as exc:
                if verbose and not quiet:
                    typer.echo(f"[WARN] Auto-publish failed: {exc}")
    

def main():
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
