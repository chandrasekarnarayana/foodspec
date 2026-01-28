"""Modeling commands: training, prediction, quality control."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from foodspec.apps.qc import apply_qc_model, train_qc_model
from foodspec.core.errors import FoodSpecValidationError
from foodspec.data.libraries import load_library
from foodspec.model_lifecycle import FrozenModel
from foodspec.model_registry import save_model as registry_save_model
from foodspec.utils.run_artifacts import get_logger, init_run_dir, write_manifest, write_run_summary

modeling_app = typer.Typer(help="Modeling commands")


def _write_qc_report(qc_result, output_dir: Path, model_type: str, threshold: float) -> Path:
    """Write QC workflow report."""
    from matplotlib import pyplot as plt

    from foodspec.reporting import (
        create_report_folder,
        save_figure,
        write_metrics_csv,
        write_summary_json,
    )

    def _to_serializable(obj):
        """Helper for JSON serialization."""
        import numpy as np

        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        return obj

    report_dir = create_report_folder(output_dir, "qc")
    scores_df = qc_result.metadata.copy()
    scores_df["score"] = qc_result.scores
    scores_df["label_pred"] = qc_result.labels_pred
    write_metrics_csv(report_dir, "scores", scores_df)

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(qc_result.scores, bins=20, alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", label="threshold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    save_figure(report_dir, "scores_hist", fig)
    plt.close(fig)

    summary = {
        "workflow": "qc",
        "model_type": model_type,
        "threshold": float(threshold),
        "counts": qc_result.labels_pred.value_counts().to_dict(),
    }
    write_summary_json(report_dir, _to_serializable(summary))
    return report_dir


@modeling_app.command("qc")
def qc_command(
    input_hdf5: str = typer.Argument(..., help="Preprocessed spectra HDF5."),
    model_type: str = typer.Option("oneclass_svm", help="QC model type: oneclass_svm or isolation_forest."),
    label_column: Optional[str] = typer.Option(None, help="Optional label column for inspection."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """Run QC/novelty detection and write report."""
    run_dir = init_run_dir(Path(output_dir))
    get_logger(run_dir)
    write_manifest(run_dir, {"command": "model.qc", "inputs": [input_hdf5], "model_type": model_type})
    try:
        ds = load_library(input_hdf5)
        model = train_qc_model(ds, train_mask=None, model_type=model_type)
        qc_result = apply_qc_model(ds, model=model, metadata=ds.metadata)
        report_dir = _write_qc_report(qc_result, run_dir, model_type=model_type, threshold=qc_result.threshold)
        write_run_summary(
            run_dir,
            {
                "status": "success",
                "qc_report": str(report_dir),
                "model_type": model_type,
                "threshold": float(qc_result.threshold),
            },
        )
        write_manifest(
            run_dir,
            {
                "command": "model.qc",
                "inputs": [input_hdf5],
                "model_type": model_type,
                "artifacts": {"qc_report": str(report_dir), "run_summary": "run_summary.json"},
            },
        )
        typer.echo(f"QC report: {report_dir}")
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@modeling_app.command("fit")
def fit_qc(
    input_hdf5: str = typer.Option(..., help="Input HDF5 library path."),
    label_col: Optional[str] = typer.Option(None, help="Label column; auto-detected if None."),
    train_label: Optional[str] = typer.Option(None, help="Train only on rows where label_col==train_label."),
    model_type: str = typer.Option("oneclass_svm", help="QC model: oneclass_svm|isolation_forest"),
    out_dir: str = typer.Option("qc_model", help="Output directory for saved model."),
):
    """Train a QC novelty detector and save model (unified CLI)."""
    run_dir = init_run_dir(Path(out_dir))
    get_logger(run_dir)
    write_manifest(run_dir, {"command": "model.fit", "inputs": [input_hdf5], "model_type": model_type})
    try:
        ds = load_library(Path(input_hdf5))
        if label_col:
            ds.label_col = label_col
        mask = None
        if train_label and ds.labels is not None:
            mask = ds.labels == train_label
        model = train_qc_model(ds, train_mask=mask, model_type=model_type)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        registry_save_model(model, Path(out_dir), name=f"qc_{model_type}")
        model_path = Path(out_dir).with_suffix(".joblib")
        meta_path = Path(out_dir).with_suffix(".json")
        write_run_summary(
            run_dir,
            {"status": "success", "model_type": model_type, "model_path": str(model_path)},
        )
        write_manifest(
            run_dir,
            {
                "command": "model.fit",
                "inputs": [input_hdf5],
                "model_type": model_type,
                "artifacts": {
                    "model": str(model_path),
                    "model_metadata": str(meta_path),
                    "run_summary": "run_summary.json",
                },
            },
        )
        print(f"Saved QC model to {out_dir}")
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@modeling_app.command("predict")
def predict(
    model: str = typer.Option(..., help="Path prefix to frozen model (.json/.pkl)."),
    input: list[str] = typer.Option(None, help="Input CSV/HDF5 files; can be provided multiple times."),
    input_dir: Optional[str] = typer.Option(None, help="Directory of inputs; used with --glob."),
    glob: str = typer.Option("*.csv", help="Glob for --input-dir."),
    output: Optional[str] = typer.Option(None, help="Output CSV for single input."),
    output_dir: Optional[str] = typer.Option(None, help="Directory for batch predictions."),
    run_dir: Optional[str] = typer.Option(None, help="Run directory for logs and summaries."),
):
    """Apply a frozen FoodSpec model to new data (unified CLI)."""
    inputs: list[Path] = []
    if input:
        inputs.extend([Path(p) for p in input])
    if input_dir:
        inputs.extend([Path(p) for p in Path(input_dir).glob(glob)])
    if not inputs:
        raise typer.BadParameter("Provide --input or --input-dir.")

    run_dir_path = init_run_dir(Path(run_dir or output_dir or "runs/predict").resolve())
    get_logger(run_dir_path)
    write_manifest(run_dir_path, {"command": "model.predict", "inputs": [str(p) for p in inputs], "model": model})

    try:
        fm = FrozenModel.load(Path(model))

        def _apply_qc(preds_df: pd.DataFrame) -> pd.DataFrame:
            if not hasattr(fm.model, "classes_"):
                return preds_df
            classes = list(fm.model.classes_)
            proba_cols = [f"proba_{cls}" for cls in classes]
            if not set(proba_cols).issubset(preds_df.columns):
                return preds_df
            from foodspec.qc import evaluate_prediction_qc

            qc_flags = []
            for _, row in preds_df.iterrows():
                probs = row[proba_cols].to_numpy(dtype=float)
                qc = evaluate_prediction_qc(probs)
                qc_flags.append(qc)
            out = preds_df.copy()
            out["qc_do_not_trust"] = [r.do_not_trust for r in qc_flags]
            out["qc_notes"] = ["; ".join(r.reasons or r.warnings) for r in qc_flags]
            return out

        if len(inputs) == 1:
            df = pd.read_csv(inputs[0])
            preds = fm.predict(df)
            preds = _apply_qc(preds)
            out_path = Path(output or run_dir_path / "predictions.csv")
            preds.to_csv(out_path, index=False)
            print(f"Predictions saved to {out_path}")
            write_run_summary(run_dir_path, {"status": "success", "rows": len(preds), "output": str(out_path)})
        else:
            out_dir_path = Path(output_dir or run_dir_path / "predictions_batch")
            out_dir_path.mkdir(parents=True, exist_ok=True)
            for inp in inputs:
                df = pd.read_csv(inp)
                preds = fm.predict(df)
                preds = _apply_qc(preds)
                out_path = out_dir_path / f"{inp.stem}_preds.csv"
                preds.to_csv(out_path, index=False)
                print(f"[{inp.name}] -> {out_path}")
            write_run_summary(run_dir_path, {"status": "success", "rows": len(inputs)})

        write_manifest(
            run_dir_path,
            {
                "command": "model.predict",
                "inputs": [str(p) for p in inputs],
                "model": model,
                "artifacts": {"run_summary": "run_summary.json"},
            },
        )
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir_path, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir_path, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)
