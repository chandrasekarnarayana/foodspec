"""Evaluation CLI command for run directories."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from sklearn.preprocessing import LabelEncoder

from foodspec.core.errors import FoodSpecValidationError
from foodspec.modeling.validation.metrics import classification_metrics_bundle
from foodspec.modeling.api import metrics_by_group
from foodspec.utils.run_artifacts import (
    get_logger,
    init_run_dir,
    safe_json_dump,
    write_manifest,
    write_run_summary,
)


def evaluate_command(
    run: Path = typer.Option(..., "--run", help="Run directory with predictions.csv."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Output directory for evaluation artifacts."),
    label_col: str = typer.Option("y_true", "--label-col", help="Label column in predictions."),
    proba_prefix: str = typer.Option("proba_", "--proba-prefix", help="Prefix for probability columns."),
    group_col: Optional[str] = typer.Option(None, "--group-col", help="Optional group column for metrics_by_group."),
):
    """Evaluate predictions from a run directory."""
    run_dir = init_run_dir(outdir or (run / "evaluation"))
    get_logger(run_dir)
    write_manifest(run_dir, {"command": "evaluate", "inputs": [run], "label_col": label_col})

    try:
        preds_path = run / "tables" / "predictions.csv"
        if not preds_path.exists():
            preds_path = run / "predictions.csv"
        if not preds_path.exists():
            raise FoodSpecValidationError(f"predictions.csv not found in {run}")

        df = pd.read_csv(preds_path)
        if label_col not in df.columns:
            raise FoodSpecValidationError(f"label_col '{label_col}' not found in predictions.")
        proba_cols = [c for c in df.columns if c.startswith(proba_prefix)]
        y_true = df[label_col].to_numpy()
        y_pred = df["y_pred"].to_numpy() if "y_pred" in df.columns else None
        if y_pred is None:
            if not proba_cols:
                raise FoodSpecValidationError("predictions must include y_pred or proba_* columns.")
            y_pred = df[proba_cols].to_numpy().argmax(axis=1)
        y_proba = df[proba_cols].to_numpy() if proba_cols else None

        encoder = LabelEncoder()
        y_true_enc = encoder.fit_transform(y_true)
        try:
            y_pred_enc = encoder.transform(y_pred)
        except Exception:
            y_pred_enc = y_pred

        bundle = classification_metrics_bundle(y_true_enc, y_pred_enc, y_proba)
        label_map = {str(i): str(lbl) for i, lbl in enumerate(encoder.classes_)}
        confusion = bundle["confusion_matrix"]
        confusion["labels"] = [label_map.get(str(l), str(l)) for l in confusion.get("labels", [])]
        per_class = {label_map.get(str(k), str(k)): v for k, v in bundle["per_class"].items()}
        bundle["confusion_matrix"] = confusion
        bundle["per_class"] = per_class

        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "metrics.json"
        safe_json_dump(
            metrics_path,
            {
                "metrics": bundle["metrics"],
                "per_class": bundle["per_class"],
                "confusion_matrix": bundle["confusion_matrix"],
            },
        )

        metrics_by_group_path = None
        if group_col and group_col in df.columns:
            by_group = metrics_by_group(
                y_true_enc,
                y_pred_enc,
                y_proba,
                df[group_col].to_numpy(),
                class_labels=encoder.classes_,
            )
            metrics_by_group_path = metrics_dir / "metrics_by_group.json"
            safe_json_dump(metrics_by_group_path, by_group)

        write_run_summary(
            run_dir,
            {
                "status": "success",
                "metrics": bundle["metrics"],
                "artifacts": {
                    "metrics": str(metrics_path),
                    "metrics_by_group": str(metrics_by_group_path) if metrics_by_group_path else None,
                },
            },
        )
        write_manifest(
            run_dir,
            {
                "command": "evaluate",
                "inputs": [run],
                "artifacts": {
                    "metrics": str(metrics_path),
                    "metrics_by_group": str(metrics_by_group_path) if metrics_by_group_path else None,
                    "run_summary": "run_summary.json",
                },
            },
        )
        typer.echo("Evaluation complete.")
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


__all__ = ["evaluate_command"]
