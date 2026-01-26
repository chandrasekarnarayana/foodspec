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
from typing import List, Optional, Union

import matplotlib
import typer

matplotlib.use("Agg")

from foodspec.cli.commands.analysis import analysis_app
from foodspec.cli.commands.data import data_app
from foodspec.cli.commands.evaluate import evaluate_command
from foodspec.cli.commands.mindmap import (
    features_app as mindmap_features_app,
    io_app as mindmap_io_app,
    model_app as mindmap_model_app,
    preprocess_app as mindmap_preprocess_app,
    qc_app as mindmap_qc_app,
    train_app as mindmap_train_app,
)
from foodspec.cli.commands.modeling import modeling_app
from foodspec.cli.commands.reporting import report_app
from foodspec.cli.commands.trust import trust_app
from foodspec.cli.commands.utils import utils_app
from foodspec.cli.commands.viz import viz_app
from foodspec.cli.commands.workflow import workflow_app
from foodspec.cli.commands.run_e2e import run_e2e_app
from foodspec.core.errors import FoodSpecQCError, FoodSpecValidationError
from foodspec.data_objects.spectral_dataset import HyperspectralDataset, SpectralDataset
from foodspec.experiment import Experiment, RunMode, ValidationScheme
from foodspec.protocol import ProtocolRunner, load_protocol, validate_protocol
import glob
import json
import pandas as pd

from foodspec.modeling.validation.quality import validate_dataset
from foodspec.qc.dataset_qc import check_class_balance
from foodspec.qc.policy import QCPolicy
from foodspec.utils.run_artifacts import (
    get_logger,
    init_run_dir,
    safe_json_dump,
    write_manifest,
    write_run_summary,
)

app = typer.Typer(help="foodspec command-line interface")

def _run_protocol_qc(cfg, input_path: Path, run_dir: Path) -> dict:
    qc_policy = QCPolicy.from_dict(getattr(cfg, "qc", {}) or {})
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
    write_run_summary(run_dir, {"qc_report": str(report_path), "qc_status": report["status"]})
    if qc_policy.required and report["status"] != "pass":
        raise FoodSpecQCError("QC required by policy; failing run.")
    return report

# Mindmap-aligned commands (new structure)
app.add_typer(mindmap_io_app, name="io")
app.add_typer(mindmap_qc_app, name="qc")
app.add_typer(mindmap_preprocess_app, name="preprocess")
app.add_typer(mindmap_features_app, name="features")
app.add_typer(run_e2e_app, name="run-e2e")
app.add_typer(mindmap_train_app, name="train")
app.add_typer(mindmap_model_app, name="model")
app.add_typer(trust_app, name="trust")
app.add_typer(report_app, name="report")
app.add_typer(viz_app, name="viz")


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
app.command("workflow-run")(workflow_app.registered_commands[0].callback)
app.command("run-exp")(workflow_app.registered_commands[1].callback)
app.command("protocol-benchmarks")(workflow_app.registered_commands[2].callback)
app.command("bench")(workflow_app.registered_commands[3].callback)

# Utilities
app.command("about")(utils_app.registered_commands[0].callback)
app.command("report-run")(utils_app.registered_commands[2].callback)
app.command("evaluate")(evaluate_command)


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
    viz: bool = typer.Option(True, "--viz/--no-viz", help="Enable/disable visualization."),
    report: bool = typer.Option(True, "--report/--no-report", help="Enable/disable reporting."),
    # New YOLO-style options (optional overrides)
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override model: lightgbm|svm|rf|logreg|plsda.",
    ),
    scheme: Optional[str] = typer.Option(
        None,
        "--scheme",
        "-s",
        help="Cross-validation scheme: loso|lobo|nested.",
    ),
    mode: Optional[str] = typer.Option(
        None,
        help="Run mode: research|regulatory|monitoring.",
    ),
    enable_trust: bool = typer.Option(
        True,
        "--trust/--no-trust",
        help="Enable/disable trust stack (calibration, conformal, abstention).",
    ),
):
    """Run a FoodSpec protocol on one or more inputs.

    CLASSIC mode (backward compatible):
        foodspec run --protocol examples/protocols/Oils.yaml \\
            --input data/oils.csv --outdir runs/exp1 --seed 0

    YOLO mode (orchestrated E2E):
        foodspec run --protocol examples/protocols/Oils.yaml \\
            --input data/oils.csv --outdir runs/exp1 \\
            --mode research --scheme lobo --model lightgbm --trust
    """
    # Check if YOLO mode is invoked (has model/scheme/mode/trust flags)
    use_yolo = any([model, scheme, mode, not enable_trust])

    if use_yolo:
        # --- YOLO mode: Use orchestration layer ---
        try:
            exp = Experiment.from_protocol(
                protocol,
                mode=mode or RunMode.RESEARCH.value,
                scheme=scheme or ValidationScheme.LOBO.value,
                model=model,
            )
            
            # Collect single input
            inputs_list: List[Path] = []
            if input:
                inputs_list.extend([Path(p) for p in input])
            if input_dir:
                inputs_list.extend([Path(p) for p in glob.glob(str(Path(input_dir) / glob_pattern))])
            
            if not inputs_list:
                raise typer.BadParameter("No inputs provided. Use --input or --input-dir.")
            
            # Run on first input (YOLO mode: single input per run)
            csv_path = inputs_list[0]
            result = exp.run(
                csv_path=csv_path,
                outdir=output_dir,
                seed=seed,
                verbose=verbose,
            )
            
            if not quiet:
                typer.echo(f"=== FoodSpec E2E Orchestration ===")
                typer.echo(f"Status: {result.status}")
                typer.echo(f"Run ID: {result.run_id}")
                if result.manifest_path:
                    typer.echo(f"Manifest: {result.manifest_path}")
                if result.report_dir:
                    typer.echo(f"Report: {result.report_dir / 'index.html'}")
                if result.summary_path:
                    typer.echo(f"Summary: {result.summary_path}")
                if result.error:
                    typer.echo(f"Error: {result.error}", err=True)
            
            raise typer.Exit(code=result.exit_code)
        
        except Exception as e:
            typer.echo(f"YOLO mode failed: {str(e)}", err=True)
            raise typer.Exit(code=3)
    
    # --- Classic mode: backward compatible ---
    # --- Classic mode: backward compatible ---
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
        run_dir = init_run_dir(Path(output_dir) / "dry_run")
        get_logger(run_dir)
        manifest_payload = {
            "command": "run",
            "inputs": inputs,
            "protocol": cfg.name,
            "dry_run": True,
            "seed": cfg.seed,
        }
        if proto_path.exists():
            manifest_payload["protocol_path"] = proto_path
        write_manifest(run_dir, manifest_payload)
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
                    write_run_summary(run_dir, {"status": "fail", "errors": diag["errors"], "dry_run": True})
                    raise typer.Exit(code=2)
                if diag["warnings"] and not quiet:
                    typer.echo("Validation warnings:")
                    for w in diag["warnings"]:
                        typer.echo(f" - {w}")
            write_run_summary(run_dir, {"status": "success", "dry_run": True})
        except typer.Exit:
            raise
        except Exception as exc:
            write_run_summary(run_dir, {"status": "fail", "error": str(exc), "dry_run": True})
            typer.echo(f"Runtime error: {exc}", err=True)
            raise typer.Exit(code=4)
        typer.echo("Dry-run complete (no execution).")
        raise typer.Exit(code=0)

    # Execute per input and save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    had_validation_errors = False
    had_runtime_errors = False
    for path in inputs:
        if verbose and not quiet:
            typer.echo(f"[INFO] Running protocol on {path}")
        target = init_run_dir(output_dir / f"{cfg.name}_{path.stem}")
        get_logger(target)
        manifest_payload = {"command": "run", "inputs": [path], "protocol": cfg.name, "seed": cfg.seed}
        if proto_path.exists():
            manifest_payload["protocol_path"] = proto_path
        write_manifest(target, manifest_payload)
        try:
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
                    write_run_summary(target, {"status": "fail", "errors": diag2["errors"]})
                    had_validation_errors = True
                    continue

            _run_protocol_qc(cfg, path, target)

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

            artifacts = {
                "report": str(target / "report.txt"),
                "metadata": str(target / "metadata.json"),
                "run_summary": "run_summary.json",
                "qc_report": str(target / "qc_report.json"),
            }
            write_run_summary(
                target,
                {
                    "status": "success",
                    "protocol": cfg.name,
                    "protocol_version": cfg.version,
                    "input": str(path),
                    "output_dir": str(target),
                    "summary": result.summary,
                    "artifacts": artifacts,
                },
            )
            write_manifest(target, {**manifest_payload, "artifacts": artifacts})

            if not quiet:
                report_html = target / "report.html"
                typer.echo(f"Run complete -> {target}")
                if report_html.exists():
                    typer.echo(f"Report: {report_html}")
            
            # Auto-generate comprehensive report if enabled
            if report:
                try:
                    from foodspec.reporting.api import build_report_from_run
                    
                    report_artifacts = build_report_from_run(
                        target,
                        out_dir=target,
                        mode=mode or "research",
                        pdf=False,
                        title=f"{cfg.name} Report",
                    )
                    
                    if not quiet:
                        typer.echo(f"Comprehensive report generated:")
                        for artifact_name, artifact_path in report_artifacts.items():
                            typer.echo(f"  - {artifact_name}: {artifact_path}")
                except Exception as e:
                    if verbose and not quiet:
                        typer.echo(f"[WARN] Report generation failed: {e}")
            
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
                        typer.echo(f"Auto-publish bundle saved -> {publish_dir}")
                except Exception as exc:
                    if verbose and not quiet:
                        typer.echo(f"[WARN] Auto-publish failed: {exc}")
        except FoodSpecQCError as exc:
            write_run_summary(target, {"status": "fail", "error": str(exc)})
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=3)
        except (FoodSpecValidationError, ValueError) as exc:
            write_run_summary(target, {"status": "fail", "error": str(exc)})
            typer.echo(f"Validation error: {exc}", err=True)
            had_validation_errors = True
        except Exception as exc:
            write_run_summary(target, {"status": "fail", "error": str(exc)})
            typer.echo(f"Runtime error: {exc}", err=True)
            had_runtime_errors = True

    if had_runtime_errors:
        raise typer.Exit(code=4)
    if had_validation_errors:
        raise typer.Exit(code=2)
    

def main():
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
