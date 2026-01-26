from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import json
import pandas as pd
import typer

from foodspec.cli.commands.modeling import predict as legacy_predict
from foodspec.cli.commands.modeling import qc_command as legacy_qc_command
from foodspec.cli.commands.preprocess import preprocess as legacy_preprocess
from foodspec.core.errors import FoodSpecQCError, FoodSpecValidationError
from foodspec.data_objects import ProtocolRunner
from foodspec.features.hybrid import extract_features
from foodspec.features.marker_panel import build_marker_panel, export_marker_panel
from foodspec.features.schema import FeatureInfo, normalize_assignment, parse_feature_config, split_spectral_dataframe
from foodspec.features.selection import feature_importance_scores, stability_selection
from foodspec.io.parsers import read_spectra
from foodspec.io.validators import validate_input
from foodspec.modeling.api import fit_predict, metrics_by_group
from foodspec.modeling.validation.quality import validate_dataset
from foodspec.qc.dataset_qc import check_class_balance, diagnose_imbalance
from foodspec.qc.engine import compute_health_scores, detect_outliers
from foodspec.qc.policy import QCPolicy
from foodspec.trust.dataset_cards import DatasetCard, write_dataset_card
from foodspec.trust.model_cards import ModelCard, write_model_card
from foodspec.utils.run_artifacts import (
    get_logger,
    init_run_dir,
    safe_json_dump,
    write_manifest,
    write_run_summary,
)

io_app = typer.Typer(help="Data extraction and validation.")
qc_app = typer.Typer(help="Quality control commands.")
preprocess_app = typer.Typer(help="Preprocessing commands.")
features_app = typer.Typer(help="Feature extraction commands.")
train_app = typer.Typer(help="Training commands.")
model_app = typer.Typer(help="Modeling commands.")


def _write_status(run_dir: Path, status: str, payload: dict) -> None:
    payload = dict(payload)
    payload["status"] = status
    write_run_summary(run_dir, payload)


def _qc_policy_from_config(cfg) -> QCPolicy:
    qc_cfg = getattr(cfg, "qc", {}) or {}
    return QCPolicy.from_dict(qc_cfg)


def _run_protocol_qc(cfg, input_path: Path, run_dir: Path) -> dict:
    qc_policy = _qc_policy_from_config(cfg)
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


def _resolve_label_col(protocol_cfg, label_col: Optional[str]) -> Optional[str]:
    if label_col:
        return label_col
    if getattr(protocol_cfg, "expected_columns", None):
        return protocol_cfg.expected_columns.get("oil_col") or protocol_cfg.expected_columns.get("label_col")
    return None


def _select_feature_columns(df: pd.DataFrame, label_col: str, group_col: Optional[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {label_col}
    if group_col:
        drop_cols.add(group_col)
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    if not feature_cols:
        raise FoodSpecValidationError("No numeric feature columns found after excluding label/group columns.")
    return feature_cols


def _raw_feature_info(feature_cols: list[str]) -> list[dict]:
    info: list[dict] = []
    for col in feature_cols:
        info.append(
            FeatureInfo(
                name=str(col),
                ftype="raw",
                assignment=normalize_assignment(None),
                description=f"Raw feature column {col}.",
                params={"column": str(col)},
            ).to_dict()
        )
    return info


def _extract_feature_table_from_csv(
    csv_path: Path,
    *,
    protocol_path: Path,
    feature_type: str,
    label_col: Optional[str],
    group_col: Optional[str],
    seed: int,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    config = parse_feature_config(protocol_path)
    feature_type = feature_type.lower()

    if feature_type == "raw":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = {c for c in [label_col, group_col] if c}
        feature_cols = [c for c in numeric_cols if c not in drop_cols]
        if not feature_cols:
            raise FoodSpecValidationError("No numeric feature columns found for raw features.")
        features_df = df[feature_cols].copy()
        meta_df = df.drop(columns=feature_cols)
        return features_df, _raw_feature_info(feature_cols), meta_df, {"feature_type": "raw"}

    if feature_type == "peaks" and not config.peaks:
        raise FoodSpecValidationError("Protocol does not define any peaks.")
    if feature_type == "bands" and not config.bands:
        raise FoodSpecValidationError("Protocol does not define any bands.")

    X, wavenumbers, meta_df = split_spectral_dataframe(df, exclude=[label_col, group_col])
    labels = None
    if label_col and label_col in meta_df.columns:
        labels = meta_df[label_col].to_numpy()

    try:
        features_df, info, details = extract_features(
            X,
            wavenumbers,
            feature_type=feature_type,
            config=config,
            labels=labels,
            seed=seed,
        )
    except ValueError as exc:
        raise FoodSpecValidationError(str(exc)) from exc
    info_payload = [entry.to_dict() if isinstance(entry, FeatureInfo) else dict(entry) for entry in info]
    return features_df, info_payload, meta_df, details


def _train_from_csv(
    protocol_path: Path,
    csv_path: Path,
    *,
    scheme: str,
    model_name: str,
    group_col: Optional[str],
    label_col: Optional[str],
    feature_type: str,
    out_dir: Path,
    unsafe_random_cv: bool,
    seed: int,
    outer_splits: int,
    inner_splits: int,
) -> None:
    run_dir = init_run_dir(out_dir)
    get_logger(run_dir)
    manifest_payload = {
        "command": "train.csv",
        "inputs": [csv_path],
        "protocol_path": protocol_path,
        "scheme": scheme,
        "model": model_name,
        "seed": seed,
        "feature_type": feature_type,
    }
    write_manifest(run_dir, manifest_payload)

    if not csv_path.exists():
        raise FoodSpecValidationError(f"CSV not found: {csv_path}")

    protocol_cfg = ProtocolRunner.from_file(protocol_path).config
    label_col = _resolve_label_col(protocol_cfg, label_col)
    if not label_col or label_col not in pd.read_csv(csv_path, nrows=1).columns:
        raise FoodSpecValidationError("Label column not found; provide --label-col or set in protocol.")

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise FoodSpecValidationError(f"Label column '{label_col}' not found in CSV.")

    groups = None
    feature_cols: list[str]
    feature_info: list[dict] | None = None
    feature_details: dict = {}
    feature_artifacts: dict[str, str] = {}

    if feature_type.lower() == "raw":
        if scheme.lower() in {"lobo", "loso"}:
            if group_col is None:
                raise FoodSpecValidationError("--group is required for LOBO/LOSO schemes.")
            if group_col not in df.columns:
                raise FoodSpecValidationError(f"Group column '{group_col}' not found in CSV.")
            groups = df[group_col].to_numpy()
        feature_cols = _select_feature_columns(df, label_col, group_col)
        X = df[feature_cols].to_numpy(dtype=float)
        y = df[label_col].to_numpy()
    else:
        features_df, feature_info, meta_df, feature_details = _extract_feature_table_from_csv(
            csv_path,
            protocol_path=protocol_path,
            feature_type=feature_type,
            label_col=label_col,
            group_col=group_col,
            seed=seed,
        )
        if label_col not in meta_df.columns:
            raise FoodSpecValidationError(f"Label column '{label_col}' not found in metadata columns.")
        y = meta_df[label_col].to_numpy()
        if scheme.lower() in {"lobo", "loso"}:
            if group_col is None:
                raise FoodSpecValidationError("--group is required for LOBO/LOSO schemes.")
            if group_col not in meta_df.columns:
                raise FoodSpecValidationError(f"Group column '{group_col}' not found in metadata columns.")
            groups = meta_df[group_col].to_numpy()
        feature_cols = list(features_df.columns)
        X = features_df.to_numpy(dtype=float)

        features_dir = run_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([meta_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        features_path = features_dir / "features.csv"
        combined.to_csv(features_path, index=False)
        if feature_info is not None:
            info_path = features_dir / "feature_info.json"
            safe_json_dump(info_path, {"features": feature_info})
            feature_artifacts = {"features": str(features_path), "feature_info": str(info_path)}
        else:
            feature_artifacts = {"features": str(features_path)}

    result = fit_predict(
        X,
        y,
        model_name=model_name,
        scheme=scheme,
        groups=groups,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        seed=seed,
        allow_random_cv=unsafe_random_cv,
    )

    models_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    validation_dir = run_dir / "validation"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("joblib is required to save models") from exc

    model_path = models_dir / "best_model.joblib"
    joblib.dump(result.model, model_path)

    model_card = {
        "model_name": model_name,
        "scheme": scheme,
        "seed": seed,
        "label_col": label_col,
        "group_col": group_col,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": feature_cols,
        "feature_type": feature_type,
        "feature_details": feature_details,
        "classes": [str(c) for c in result.classes],
        "metrics": result.metrics,
        "metrics_ci": result.metrics_ci,
        "best_params": result.best_params,
        "protocol_path": str(protocol_path),
        "warnings": ["unsafe_random_cv"] if unsafe_random_cv and scheme.lower() in {"random", "kfold"} else [],
    }
    model_card_path = models_dir / "model_card.json"
    safe_json_dump(model_card_path, model_card)

    folds_path = validation_dir / "folds.json"
    safe_json_dump(folds_path, {"folds": result.folds})

    metrics_path = metrics_dir / "metrics.json"
    safe_json_dump(
        metrics_path,
        {
            "metrics": result.metrics,
            "metrics_ci": result.metrics_ci,
            "per_class": result.per_class,
            "confusion_matrix": result.confusion_matrix,
        },
    )

    metrics_by_group_path = metrics_dir / "metrics_by_group.json"
    if result.groups is not None:
        by_group = metrics_by_group(
            result.y_true,
            result.y_pred,
            result.y_proba,
            result.groups,
            class_labels=result.classes,
        )
        safe_json_dump(metrics_by_group_path, by_group)
    else:
        safe_json_dump(metrics_by_group_path, {})

    write_run_summary(
        run_dir,
        {
            "status": "success",
            "metrics": result.metrics,
            "feature_type": feature_type,
            "artifacts": {
                "model": str(model_path),
                "model_card": str(model_card_path),
                "folds": str(folds_path),
                "metrics": str(metrics_path),
                "metrics_by_group": str(metrics_by_group_path),
                **feature_artifacts,
            },
        },
    )
    write_manifest(
        run_dir,
        {
            **manifest_payload,
            "artifacts": {
                "model": str(model_path),
                "model_card": str(model_card_path),
                "folds": str(folds_path),
                "metrics": str(metrics_path),
                "metrics_by_group": str(metrics_by_group_path),
                **feature_artifacts,
                "run_summary": "run_summary.json",
            },
        },
    )


def _summarize_dataset_for_card(input_path: Path, label_col: Optional[str]) -> dict:
    summary = {"size": {}, "labels": {}, "features": []}
    if input_path.suffix.lower() == ".csv" and input_path.exists():
        df = pd.read_csv(input_path)
        summary["size"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        summary["features"] = list(df.columns)
        if label_col and label_col in df.columns:
            counts = df[label_col].value_counts(dropna=False).to_dict()
            summary["labels"] = {"label_col": label_col, "class_counts": counts}
    return summary


@io_app.command("validate")
def io_validate(
    path: Path = typer.Argument(..., help="Input file or directory."),
    run_dir: Path = typer.Option(Path("runs/io_validate"), help="Run output directory."),
):
    """Validate an input path and inferred format."""

    run_dir = init_run_dir(run_dir)
    get_logger(run_dir)
    write_manifest(
        run_dir,
        {"command": "io.validate", "inputs": [path]},
    )
    try:
        results = validate_input(str(path), deep=True)
        output_path = run_dir / "io_validation.json"
        safe_json_dump(output_path, results)
        if results["errors"]:
            _write_status(run_dir, "fail", {"errors": results["errors"], "warnings": results["warnings"]})
            for err in results["errors"]:
                typer.echo(f"ERROR: {err}", err=True)
            raise typer.Exit(code=2)
        _write_status(run_dir, "success", {"warnings": results["warnings"], "io_validation": str(output_path)})
        typer.echo("IO validation complete.")
    except FoodSpecValidationError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)

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
    required: bool = typer.Option(False, "--required/--no-required", help="Fail run when QC fails."),
):
    """Run spectral QC diagnostics."""
    run_dir = init_run_dir(run_dir)
    get_logger(run_dir)
    policy = QCPolicy(required=required)
    write_manifest(
        run_dir,
        {"command": "qc.spectral", "inputs": [path], "outlier_method": outlier_method},
    )
    try:
        fs = read_spectra(path)
        health = compute_health_scores(fs)
        outliers = detect_outliers(fs, method=outlier_method)
        policy_result = policy.evaluate_spectrum(health, outliers)
        payload = {
            "health": health.table.to_dict(orient="list"),
            "health_aggregates": health.aggregates,
            "outliers": {
                "labels": outliers.labels.tolist(),
                "scores": outliers.scores.tolist(),
                "method": outliers.method,
            },
            "policy_result": policy_result,
        }
        output_path = run_dir / "qc_report.json"
        safe_json_dump(output_path, payload)
        _write_status(
            run_dir,
            "success" if policy_result["status"] == "pass" else "fail",
            {"qc": "spectral", "samples": len(outliers.labels), "qc_report": str(output_path)},
        )
        if policy.required and policy_result["status"] != "pass":
            raise FoodSpecQCError("QC required by policy; failing run.")
        typer.echo("Spectral QC complete.")
    except FoodSpecQCError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3)
    except FoodSpecValidationError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@qc_app.command("dataset")
def qc_dataset(
    csv_path: Path = typer.Argument(..., help="CSV with metadata and labels."),
    label_column: str = typer.Option(..., help="Label column."),
    batch_column: Optional[str] = typer.Option(None, help="Optional batch column."),
    run_dir: Path = typer.Option(Path("runs/qc_dataset"), help="Run output directory."),
    required: bool = typer.Option(False, "--required/--no-required", help="Fail run when QC fails."),
):
    """Run dataset QC diagnostics."""
    run_dir = init_run_dir(run_dir)
    get_logger(run_dir)
    policy = QCPolicy(required=required)
    write_manifest(
        run_dir,
        {
            "command": "qc.dataset",
            "inputs": [csv_path],
            "label_column": label_column,
            "batch_column": batch_column,
        },
    )
    try:
        df = pd.read_csv(csv_path)
        balance = check_class_balance(df, label_column)
        dummy = df.copy()
        dummy_x = [[0.0] * 2 for _ in range(len(dummy))]
        from foodspec.data_objects.spectra_set import FoodSpectrumSet

        ds = FoodSpectrumSet(x=dummy_x, wavenumbers=[1.0, 2.0], metadata=dummy, modality="raman")
        diagnostics = diagnose_imbalance(ds, label_column, stratification_column=batch_column)
        policy_result = policy.evaluate_dataset(balance)
        payload = {"balance": balance, "diagnostics": diagnostics, "policy_result": policy_result}
        output_path = run_dir / "qc_report.json"
        safe_json_dump(output_path, payload)
        _write_status(
            run_dir,
            "success" if policy_result["status"] == "pass" else "fail",
            {"qc": "dataset", "rows": len(df), "qc_report": str(output_path)},
        )
        if policy.required and policy_result["status"] != "pass":
            raise FoodSpecQCError("QC required by policy; failing run.")
        typer.echo("Dataset QC complete.")
    except FoodSpecQCError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3)
    except FoodSpecValidationError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)

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
    run_dir = init_run_dir(run_dir)
    get_logger(run_dir)
    manifest_payload = {
        "command": "preprocess.run",
        "inputs": [input_path],
        "protocol_path": protocol_path,
    }
    write_manifest(run_dir, manifest_payload)
    try:
        runner = ProtocolRunner.from_file(protocol_path)
        full_steps = list(runner.config.steps)
        _run_protocol_qc(runner.config, input_path, run_dir)
        runner.config.steps = [s for s in full_steps if s.get("type") == "preprocess"]
        if not runner.config.steps:
            raise typer.BadParameter("Protocol has no preprocess steps.")
        result = runner.run([input_path])
        _write_status(run_dir, "success", {"logs": result.logs, "summary": result.summary})
        write_manifest(
            run_dir,
            {
                **manifest_payload,
                "seed": runner.config.seed,
                "artifacts": {"run_summary": "run_summary.json", "qc_report": "qc_report.json"},
            },
        )
        typer.echo("Preprocess run complete.")
    except FoodSpecQCError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3)
    except FoodSpecValidationError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@features_app.command("extract")
def features_extract(
    protocol_path: Path = typer.Option(..., "--protocol", "-p", help="Protocol YAML/JSON file."),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Input CSV or HDF5."),
    csv_path: Optional[Path] = typer.Option(None, "--csv", help="CSV file with spectra and metadata."),
    feature_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Feature backend: raw|peaks|bands|pca|pls|hybrid.",
    ),
    label_col: Optional[str] = typer.Option(None, "--label-col", help="Label column override (if present)."),
    group_col: Optional[str] = typer.Option(None, "--group", help="Group column override (if present)."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    run_dir: Path = typer.Option(Path("runs/features"), "--run-dir", "--outdir", help="Run output directory."),
):
    """Extract features using a protocol."""
    run_dir = init_run_dir(run_dir)
    get_logger(run_dir)
    manifest_payload = {"command": "features.extract", "inputs": [], "protocol_path": protocol_path}
    if csv_path:
        manifest_payload["inputs"] = [csv_path]
    elif input_path:
        manifest_payload["inputs"] = [input_path]
    write_manifest(run_dir, manifest_payload)
    try:
        if feature_type:
            data_path = csv_path or input_path
            if data_path is None:
                raise typer.BadParameter("Provide --csv or --input for feature extraction.")
            protocol_cfg = ProtocolRunner.from_file(protocol_path).config
            resolved_label = _resolve_label_col(protocol_cfg, label_col)
            features_df, info_payload, meta_df, details = _extract_feature_table_from_csv(
                data_path,
                protocol_path=protocol_path,
                feature_type=feature_type,
                label_col=resolved_label,
                group_col=group_col,
                seed=seed,
            )
            features_dir = run_dir / "features"
            features_dir.mkdir(parents=True, exist_ok=True)
            combined = pd.concat([meta_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
            features_path = features_dir / "features.csv"
            combined.to_csv(features_path, index=False)
            info_path = features_dir / "feature_info.json"
            safe_json_dump(info_path, {"features": info_payload})
            _write_status(
                run_dir,
                "success",
                {
                    "features": str(features_path),
                    "feature_info": str(info_path),
                    "feature_type": feature_type,
                    "feature_details": details,
                },
            )
            write_manifest(
                run_dir,
                {
                    **manifest_payload,
                    "seed": seed,
                    "feature_type": feature_type,
                    "artifacts": {
                        "features": str(features_path),
                        "feature_info": str(info_path),
                        "run_summary": "run_summary.json",
                    },
                },
            )
            typer.echo("Feature extraction complete.")
        else:
            if input_path is None:
                raise typer.BadParameter("--input is required when --type is not provided.")
            runner = ProtocolRunner.from_file(protocol_path)
            full_steps = list(runner.config.steps)
            _run_protocol_qc(runner.config, input_path, run_dir)
            runner.config.steps = [s for s in full_steps if s.get("type") in {"preprocess", "rq_analysis"}]
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
            _write_status(run_dir, "success", {"logs": result.logs, "summary": result.summary})
            write_manifest(
                run_dir,
                {
                    **manifest_payload,
                    "seed": runner.config.seed,
                    "artifacts": {
                        "tables": str(tables_dir),
                        "qc_report": "qc_report.json",
                        "run_summary": "run_summary.json",
                    },
                },
            )
            typer.echo("Feature extraction complete.")
    except FoodSpecQCError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3)
    except FoodSpecValidationError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@features_app.command("select")
def features_select(
    run_path: Path = typer.Option(..., "--run", help="Run directory containing features."),
    out_dir: Path = typer.Option(Path("runs/features_select"), "--run-dir", "--outdir", help="Run output directory."),
    method: str = typer.Option("stability", "--method", help="Selection method (stability)."),
    k: int = typer.Option(8, "--k", help="Number of features in the marker panel."),
    label_col: Optional[str] = typer.Option(None, "--label-col", help="Label column override."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    n_resamples: int = typer.Option(30, "--n-resamples", help="Resamples for stability selection."),
    sample_fraction: float = typer.Option(0.75, "--sample-fraction", help="Sample fraction for subsampling."),
):
    """Select a minimal marker panel from a prior feature run."""

    run_dir = init_run_dir(out_dir)
    get_logger(run_dir)
    manifest_payload = {
        "command": "features.select",
        "inputs": [run_path],
        "method": method,
        "k": k,
        "seed": seed,
    }
    write_manifest(run_dir, manifest_payload)

    try:
        features_path = run_path / "features" / "features.csv"
        feature_info_path = run_path / "features" / "feature_info.json"
        if not features_path.exists():
            summary_path = run_path / "run_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    candidate = summary.get("features")
                    if candidate:
                        features_path = Path(candidate)
                except Exception:
                    pass
        if not features_path.exists():
            raise FoodSpecValidationError("Could not locate features.csv in the run directory.")

        df = pd.read_csv(features_path)
        resolved_label = label_col
        manifest_path = run_path / "manifest.json"
        if resolved_label is None and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                protocol_path = manifest.get("protocol_path")
                if protocol_path:
                    cfg = ProtocolRunner.from_file(protocol_path).config
                    resolved_label = _resolve_label_col(cfg, None)
            except Exception:
                resolved_label = label_col
        if resolved_label is None or resolved_label not in df.columns:
            raise FoodSpecValidationError("Label column not found; provide --label-col.")

        if method.lower() != "stability":
            raise FoodSpecValidationError(f"Unsupported selection method: {method}")

        feature_info = None
        if feature_info_path.exists():
            try:
                feature_info = json.loads(feature_info_path.read_text()).get("features")
            except Exception:
                feature_info = None

        if feature_info:
            feature_cols = [entry.get("name") for entry in feature_info if entry.get("name") in df.columns]
        else:
            feature_cols = _select_feature_columns(df, resolved_label, None)
        if not feature_cols:
            raise FoodSpecValidationError("No feature columns found for selection.")

        X = df[feature_cols]
        y = df[resolved_label].to_numpy()

        stability_df = stability_selection(
            X,
            y,
            n_resamples=n_resamples,
            sample_fraction=sample_fraction,
            seed=seed,
        )
        perf_scores = feature_importance_scores(X, y)

        panel = build_marker_panel(
            stability_df,
            k=k,
            performance=perf_scores,
            feature_info=feature_info,
        )

        features_dir = run_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        stability_path = features_dir / "stability.csv"
        stability_df.to_csv(stability_path, index=False)
        panel_json, panel_csv = export_marker_panel(panel, features_dir)

        _write_status(
            run_dir,
            "success",
            {
                "feature_panel": str(panel_json),
                "stability": str(stability_path),
                "features": str(features_path),
            },
        )
        write_manifest(
            run_dir,
            {
                **manifest_payload,
                "artifacts": {
                    "marker_panel_json": str(panel_json),
                    "marker_panel_csv": str(panel_csv),
                    "stability": str(stability_path),
                    "run_summary": "run_summary.json",
                },
            },
        )
        typer.echo("Feature selection complete.")
    except FoodSpecValidationError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@train_app.command("run")
def train_run(
    protocol_path: Path = typer.Option(..., "--protocol", "-p", help="Protocol YAML/JSON file."),
    input_path: Path = typer.Option(..., "--input", "-i", help="Input CSV or HDF5."),
    run_dir: Path = typer.Option(Path("runs/train"), "--run-dir", "--outdir", help="Run output directory."),
):
    """Run a training protocol (full execution)."""
    run_dir = init_run_dir(run_dir)
    get_logger(run_dir)
    manifest_payload = {
        "command": "train.run",
        "inputs": [input_path],
        "protocol_path": protocol_path,
    }
    write_manifest(run_dir, manifest_payload)
    try:
        runner = ProtocolRunner.from_file(protocol_path)
        qc_report = _run_protocol_qc(runner.config, input_path, run_dir)
        result = runner.run([input_path])
        _write_status(run_dir, "success", {"logs": result.logs, "summary": result.summary})
        manifest_data: dict = {}
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text())
            except Exception:
                manifest_data = {}

        label_col = None
        if runner.config.expected_columns:
            label_col = runner.config.expected_columns.get("oil_col") or runner.config.expected_columns.get("label_col")
        dataset_summary = _summarize_dataset_for_card(input_path, label_col)
        model_card = ModelCard(
            name=runner.config.name,
            version=runner.config.version,
            overview=runner.config.description or "Model trained via FoodSpec protocol.",
            intended_use=runner.config.when_to_use or "Spectral classification or regression for food matrices.",
            non_goals=[
                "Not a clinical or regulatory device.",
                "Not a vendor replacement tool.",
            ],
            limitations=[
                "Performance depends on representative spectra and labels.",
                "Out-of-distribution instruments or matrices may reduce accuracy.",
            ],
            training_data={
                "inputs": [str(input_path)],
                "protocol_path": str(protocol_path),
            },
            metrics={"summary": result.summary},
            qc={
                "status": qc_report.get("status"),
                "policy_result": qc_report.get("policy_result"),
            },
            reproducibility={
                "foodspec_version": manifest_data.get("foodspec_version"),
                "python_version": manifest_data.get("python_version"),
                "platform": manifest_data.get("platform"),
                "git_commit": manifest_data.get("git_commit"),
                "seed": runner.config.seed,
            },
            failure_modes=[
                "Low signal-to-noise spectra may yield unstable predictions.",
                "Missing QC checks can allow degraded data into the model.",
            ],
        )
        model_card_path = write_model_card(run_dir, model_card, format="md")

        dataset_card = DatasetCard(
            name=input_path.stem,
            description=runner.config.description or "Dataset used to train the protocol.",
            collection="protocol_input",
            features=dataset_summary["features"],
            size=dataset_summary["size"],
            labels=dataset_summary["labels"],
            provenance={"inputs": [str(input_path)], "protocol_path": str(protocol_path)},
            qc_summary={
                "status": qc_report.get("status"),
                "policy_result": qc_report.get("policy_result"),
            },
            limitations=[
                "Label quality and class balance affect downstream performance.",
            ],
        )
        dataset_card_path = write_dataset_card(run_dir, dataset_card, format="md")
        write_run_summary(
            run_dir,
            {
                "model_card": str(model_card_path),
                "dataset_card": str(dataset_card_path),
            },
        )
        write_manifest(
            run_dir,
            {
                **manifest_payload,
                "seed": runner.config.seed,
                "artifacts": {
                    "run_summary": "run_summary.json",
                    "qc_report": "qc_report.json",
                    "model_card": str(model_card_path),
                    "dataset_card": str(dataset_card_path),
                },
            },
        )
        typer.echo("Training run complete.")
    except FoodSpecQCError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3)
    except FoodSpecValidationError as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        _write_status(run_dir, "fail", {"error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@train_app.callback(invoke_without_command=True)
def train_root(
    ctx: typer.Context,
    protocol_path: Optional[Path] = typer.Option(None, "--protocol", "-p", help="Protocol YAML/JSON file."),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Input CSV or HDF5."),
    csv_path: Optional[Path] = typer.Option(None, "--csv", help="CSV file with features and labels."),
    scheme: str = typer.Option("nested", "--scheme", help="Validation scheme: nested|lobo|loso|random."),
    group: Optional[str] = typer.Option(None, "--group", help="Group column for LOBO/LOSO."),
    model: str = typer.Option("logreg", "--model", help="Model name (logreg, svm_linear, svm_rbf, rf, pls_da, xgb, lgbm)."),
    label_col: Optional[str] = typer.Option(None, "--label-col", help="Label column in CSV (overrides protocol)."),
    features: str = typer.Option(
        "raw",
        "--features",
        help="Feature backend: raw|peaks|bands|pca|pls|hybrid.",
    ),
    unsafe_random_cv: bool = typer.Option(
        False,
        "--unsafe-random-cv",
        help="Allow random CV for real food data (logs warning).",
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    outer_splits: int = typer.Option(5, "--outer-splits", help="Outer CV folds."),
    inner_splits: int = typer.Option(3, "--inner-splits", help="Inner CV folds."),
    run_dir: Path = typer.Option(Path("runs/train"), help="Run output directory."),
):
    """Train group (mindmap command alias)."""

    if ctx.invoked_subcommand is None:
        if csv_path is not None:
            if protocol_path is None:
                raise typer.BadParameter("--protocol is required for CSV training.")
            if unsafe_random_cv:
                typer.echo("Warning: using random CV on food data (unsafe).", err=True)
            try:
                _train_from_csv(
                    protocol_path,
                    csv_path,
                    scheme=scheme,
                    model_name=model,
                    group_col=group,
                    label_col=label_col,
                    feature_type=features,
                    out_dir=run_dir,
                    unsafe_random_cv=unsafe_random_cv,
                    seed=seed,
                    outer_splits=outer_splits,
                    inner_splits=inner_splits,
                )
                return
            except FoodSpecValidationError as exc:
                write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
                typer.echo(f"Validation error: {exc}", err=True)
                raise typer.Exit(code=2)
            except Exception as exc:
                write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
                typer.echo(f"Runtime error: {exc}", err=True)
                raise typer.Exit(code=4)
        if protocol_path is None or input_path is None:
            raise typer.BadParameter("--protocol and --input are required.")
        train_run(protocol_path, input_path, run_dir)


@model_app.callback(invoke_without_command=True)
def model_root(
    ctx: typer.Context,
    protocol_path: Optional[Path] = typer.Option(None, "--protocol", "-p", help="Protocol YAML/JSON file."),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Input CSV or HDF5."),
    run_dir: Path = typer.Option(Path("runs/model"), help="Run output directory."),
):
    """Model group entry (alias to train run)."""

    if ctx.invoked_subcommand is None:
        if protocol_path is None or input_path is None:
            raise typer.BadParameter("--protocol and --input are required.")
        train_run(protocol_path, input_path, run_dir)


model_app.command("run")(train_run)
model_app.command("predict")(legacy_predict)
