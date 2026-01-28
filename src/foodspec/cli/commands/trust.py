"""Trust and uncertainty CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from foodspec.core.errors import FoodSpecValidationError
from foodspec.trust.abstention import apply_abstention_rules
from foodspec.trust.calibration import IsotonicCalibrator, PlattCalibrator
from foodspec.trust.conformal import MondrianConformalClassifier
from foodspec.trust.coverage import coverage_by_group
from foodspec.trust.metrics import bootstrap_coverage_efficiency, compute_calibration_metrics
from foodspec.trust.readiness import evaluate_run_readiness, load_trust_payload
from foodspec.trust.schema import AbstentionArtifact, CalibrationArtifact, ConformalArtifact, ReadinessArtifact
from foodspec.utils.run_artifacts import (
    get_logger,
    init_run_dir,
    safe_json_dump,
    update_manifest,
    write_manifest,
    write_run_summary,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

trust_app = typer.Typer(help="Trust & uncertainty commands.")


def _select_proba_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise FoodSpecValidationError(f"No probability columns found with prefix '{prefix}'.")
    return cols


def _load_predictions(run_dir: Path, label_col: str, proba_prefix: str) -> pd.DataFrame:
    table_path = run_dir / "tables" / "predictions.csv"
    if not table_path.exists():
        table_path = run_dir / "predictions.csv"
    if not table_path.exists():
        raise FoodSpecValidationError(f"predictions.csv not found in {run_dir}")
    df = pd.read_csv(table_path)
    if label_col not in df.columns:
        raise FoodSpecValidationError(f"label_col '{label_col}' not found in predictions.csv")
    _select_proba_columns(df, proba_prefix)
    return df


def _split_calibration_test(
    df: pd.DataFrame,
    label_col: str,
    *,
    split_col: str | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    split_col = split_col or ("split" if "split" in df.columns else None)
    if split_col and split_col in df.columns:
        split_vals = df[split_col].astype(str).str.lower()
        cal_mask = split_vals.isin({"cal", "calibration", "calib"})
        test_mask = split_vals.isin({"test", "validation", "val"})
        if cal_mask.any() and test_mask.any():
            return df[cal_mask].copy(), df[test_mask].copy(), warnings
        warnings.append("split column present but missing calibration/test labels; falling back to random split")

    # Fallback: stratified split
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=seed)
        y = df[label_col].astype(int).to_numpy()
        train_idx, test_idx = next(splitter.split(df, y))
        cal_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        if cal_df.empty or test_df.empty:
            raise FoodSpecValidationError("Calibration/test split is empty; provide a valid split column.")
        warnings.append("No explicit calibration split found; used stratified random split.")
        return cal_df, test_df, warnings
    except Exception as exc:
        raise FoodSpecValidationError(
            "Unable to create a held-out calibration split. Provide a split column "
            "with calibration/test labels or more data."
        ) from exc


def _trust_dir(run_dir: Path) -> Path:
    trust_dir = run_dir / "trust"
    trust_dir.mkdir(parents=True, exist_ok=True)
    return trust_dir


def _update_trust_outputs(run_dir: Path, payload: dict) -> Path:
    path = run_dir / "trust_outputs.json"
    existing = load_trust_payload(path)
    existing.update(payload)
    safe_json_dump(path, existing)
    return path


@trust_app.command("fit")
def fit_calibration(
    run: Path = typer.Option(..., "--run", help="Run directory with predictions.csv."),
    method: str = typer.Option("platt", help="Calibration method: platt|isotonic."),
    label_col: str = typer.Option("y_true", help="Column containing true labels."),
    proba_prefix: str = typer.Option("proba_", help="Prefix for probability columns."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Output directory for trust artifacts."),
    seed: int = typer.Option(0, "--seed", help="Random seed for splits."),
):
    """Fit calibration on a held-out calibration split."""
    run_dir = init_run_dir(outdir or run)
    get_logger(run_dir)
    update_manifest(run_dir, {"command": "trust.fit", "inputs": [run], "method": method})
    try:
        df = _load_predictions(run, label_col, proba_prefix)
        cal_df, test_df, warnings = _split_calibration_test(df, label_col, seed=seed)
        proba_cols = _select_proba_columns(df, proba_prefix)

        y_cal = cal_df[label_col].to_numpy(dtype=int)
        proba_cal = cal_df[proba_cols].to_numpy(dtype=float)
        y_test = test_df[label_col].to_numpy(dtype=int)
        proba_test = test_df[proba_cols].to_numpy(dtype=float)

        if method == "platt":
            calibrator = PlattCalibrator()
        elif method == "isotonic":
            calibrator = IsotonicCalibrator()
        else:
            raise FoodSpecValidationError("method must be 'platt' or 'isotonic'")

        calibrator.fit(y_cal, proba_cal)
        proba_calibrated = calibrator.transform(proba_test)

        metrics_before = compute_calibration_metrics(y_test, proba_test)
        metrics_after = compute_calibration_metrics(y_test, proba_calibrated)

        trust_dir = _trust_dir(run_dir)
        model_path = trust_dir / "calibration_model.joblib"
        if joblib is None:
            raise RuntimeError("joblib is required to save calibration model")
        joblib.dump(calibrator, model_path)

        artifact = CalibrationArtifact(
            method=method,
            model_path=str(model_path),
            n_calibration=len(cal_df),
            n_test=len(test_df),
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            warnings=warnings,
        )
        calib_path = trust_dir / "calibration.json"
        safe_json_dump(calib_path, artifact.to_dict())

        trust_outputs = _update_trust_outputs(
            run_dir,
            {
                "calibration": artifact.to_dict(),
                "ece": metrics_after.get("ece"),
                "brier": metrics_after.get("brier"),
                "nll": metrics_after.get("nll"),
            },
        )
        readiness = evaluate_run_readiness(run, load_trust_payload(trust_outputs))
        readiness_artifact = ReadinessArtifact(
            score=readiness.score, components=readiness.components, notes=readiness.notes
        )
        readiness_path = trust_dir / "readiness.json"
        safe_json_dump(readiness_path, readiness_artifact.to_dict())
        _update_trust_outputs(run_dir, {"readiness": readiness_artifact.to_dict()})

        write_run_summary(
            run_dir,
            {
                "status": "success",
                "trust_outputs": str(trust_outputs),
                "artifacts": {
                    "calibration": str(calib_path),
                    "readiness": str(readiness_path),
                    "trust_outputs": str(trust_outputs),
                },
            },
        )
        update_manifest(run_dir, {"artifacts": {"trust_outputs": str(trust_outputs)}})
        typer.echo("Calibration artifacts written.")
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@trust_app.command("calibrate")
def calibrate_compat(
    run: Optional[Path] = typer.Option(None, "--run", help="Run directory with predictions.csv."),
    calibration_csv: Optional[Path] = typer.Option(
        None, "--calibration-csv", "-c", help="CSV with labels and probabilities."
    ),
    label_col: str = typer.Option("y_true", help="Column containing true labels."),
    proba_prefix: str = typer.Option("proba_", help="Prefix for probability columns."),
    method: str = typer.Option("platt", help="Calibration method: platt|isotonic."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Output directory for trust artifacts."),
    seed: int = typer.Option(0, "--seed", help="Random seed for splits."),
):
    """Deprecated alias for calibration (use `trust fit` or `trust calibrate-csv`)."""
    try:
        if run and calibration_csv:
            raise FoodSpecValidationError("Provide either --run or --calibration-csv, not both.")
        if run:
            typer.echo("Deprecated: use `foodspec trust fit`.", err=True)
            return fit_calibration(
                run=run,
                method=method,
                label_col=label_col,
                proba_prefix=proba_prefix,
                outdir=outdir,
                seed=seed,
            )
        if calibration_csv:
            typer.echo("Deprecated: use `foodspec trust calibrate-csv`.", err=True)
            return calibrate_csv(
                calibration_csv=calibration_csv,
                label_col=label_col,
                proba_prefix=proba_prefix,
                method=method,
                run_dir=outdir or Path("runs/trust_calibration"),
            )
        raise FoodSpecValidationError("Provide --run or --calibration-csv for calibration.")
    except FoodSpecValidationError as exc:
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)


@trust_app.command("conformal")
def conformal_from_run(
    run: Optional[Path] = typer.Option(None, "--run", help="Run directory with predictions.csv."),
    calibration_csv: Optional[Path] = typer.Option(None, "--calibration-csv", "-c", help="Legacy calibration CSV."),
    test_csv: Optional[Path] = typer.Option(None, "--test-csv", "-t", help="Legacy test CSV."),
    alpha: float = typer.Option(0.1, help="Significance level (1 - target coverage)."),
    mondrian: Optional[str] = typer.Option(None, "--mondrian", help="Mondrian conditioning, e.g. group=stage."),
    condition_col: Optional[str] = typer.Option(
        None, "--condition-col", help="Legacy conditioning column for CSV mode."
    ),
    label_col: str = typer.Option("y_true", help="Column containing true labels."),
    proba_prefix: str = typer.Option("proba_", help="Prefix for probability columns."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Output directory for trust artifacts."),
    seed: int = typer.Option(0, "--seed", help="Random seed for splits."),
):
    """Generate conformal prediction sets from run predictions."""
    try:
        if run is None:
            if calibration_csv and test_csv:
                typer.echo("Deprecated: use `foodspec trust conformal-csv`.", err=True)
                return conformal_csv(
                    calibration_csv=calibration_csv,
                    test_csv=test_csv,
                    label_col=label_col,
                    proba_prefix=proba_prefix,
                    alpha=alpha,
                    condition_col=condition_col,
                    run_dir=outdir or Path("runs/trust_conformal"),
                )
            raise FoodSpecValidationError("Provide --run or both --calibration-csv and --test-csv.")

        run_dir = init_run_dir(outdir or run)
        get_logger(run_dir)
        update_manifest(run_dir, {"command": "trust.conformal", "inputs": [run], "alpha": alpha})

        df = _load_predictions(run, label_col, proba_prefix)
        condition_override = condition_col
        condition_col = None
        warnings: list[str] = []
        if mondrian:
            parts = mondrian.split("=", 1)
            if len(parts) == 2 and parts[0].strip() == "group":
                condition_col = parts[1].strip()
            else:
                warnings.append("Invalid mondrian format; expected group=<column>. Using global coverage.")
        elif condition_override:
            condition_col = condition_override
        if condition_col and condition_col not in df.columns:
            warnings.append(f"Mondrian column '{condition_col}' not found; using global coverage.")
            condition_col = None
        cal_df, test_df, split_warnings = _split_calibration_test(df, label_col, seed=seed)
        warnings.extend(split_warnings)
        proba_cols = _select_proba_columns(df, proba_prefix)

        y_cal = cal_df[label_col].to_numpy(dtype=int)
        proba_cal = cal_df[proba_cols].to_numpy(dtype=float)
        meta_cal = cal_df[condition_col].to_numpy() if condition_col and condition_col in cal_df.columns else None

        cp = MondrianConformalClassifier(alpha=alpha, condition_key=condition_col)
        cp.fit(y_cal, proba_cal, meta_cal=meta_cal)

        proba_test = test_df[proba_cols].to_numpy(dtype=float)
        meta_test = test_df[condition_col].to_numpy() if condition_col and condition_col in test_df.columns else None
        y_true_test = test_df[label_col].to_numpy(dtype=int)

        result = cp.predict_sets(proba_test, meta_test=meta_test, y_true=y_true_test)
        covered = np.array([int(y_true_test[i] in result.prediction_sets[i]) for i in range(len(y_true_test))])
        coverage_stats = bootstrap_coverage_efficiency(covered, np.asarray(result.set_sizes), seed=seed)

        per_group = None
        if meta_test is not None:
            df_sets = result.to_dataframe(y_true=y_true_test, bin_values=meta_test)
            group_table = coverage_by_group(df_sets, group_col="bin")
            per_group = group_table.to_dict(orient="records")
            small_groups = group_table[group_table["n_samples"] < cp.min_bin_size]
            if not small_groups.empty:
                warnings.append("Some groups are small; coverage aggregated with global threshold.")

        artifact = ConformalArtifact(
            alpha=alpha,
            condition_key=condition_col,
            coverage=result.coverage,
            mean_set_size=float(np.mean(result.set_sizes)) if result.set_sizes else 0.0,
            coverage_ci=coverage_stats["coverage_ci"],
            efficiency=coverage_stats["efficiency"],
            efficiency_ci=coverage_stats["efficiency_ci"],
            coverage_curve=coverage_stats["curve"],
            per_group=per_group,
            warnings=warnings,
        )

        trust_dir = _trust_dir(run_dir)
        sets_path = trust_dir / "conformal_sets.csv"
        result.to_dataframe(y_true=y_true_test, bin_values=meta_test).to_csv(sets_path, index=False)
        conformal_path = trust_dir / "conformal.json"
        payload = artifact.to_dict()
        payload.update(
            {
                "prediction_sets": result.prediction_sets,
                "set_sizes": result.set_sizes,
                "thresholds": result.thresholds,
            }
        )
        safe_json_dump(conformal_path, payload)

        trust_outputs = _update_trust_outputs(
            run_dir,
            {"conformal": payload, "coverage": artifact.coverage, "mean_set_size": artifact.mean_set_size},
        )
        readiness = evaluate_run_readiness(run, load_trust_payload(trust_outputs))
        readiness_artifact = ReadinessArtifact(
            score=readiness.score, components=readiness.components, notes=readiness.notes
        )
        readiness_path = trust_dir / "readiness.json"
        safe_json_dump(readiness_path, readiness_artifact.to_dict())
        _update_trust_outputs(run_dir, {"readiness": readiness_artifact.to_dict()})

        write_run_summary(
            run_dir,
            {
                "status": "success",
                "trust_outputs": str(trust_outputs),
                "artifacts": {
                    "conformal": str(conformal_path),
                    "conformal_sets": str(sets_path),
                    "readiness": str(readiness_path),
                    "trust_outputs": str(trust_outputs),
                },
            },
        )
        update_manifest(run_dir, {"artifacts": {"trust_outputs": str(trust_outputs)}})
        typer.echo("Conformal artifacts written.")
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@trust_app.command("abstain")
def abstain_from_run(
    run: Path = typer.Option(..., "--run", help="Run directory with predictions.csv."),
    tau: float = typer.Option(0.7, "--tau", help="Reject if max probability < tau."),
    max_set_size: Optional[int] = typer.Option(2, "--max-set-size", help="Reject if conformal set > k."),
    label_col: str = typer.Option("y_true", help="Column containing true labels."),
    proba_prefix: str = typer.Option("proba_", help="Prefix for probability columns."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Output directory for trust artifacts."),
    density_col: Optional[str] = typer.Option(None, "--density-col", help="Optional density score column."),
    density_quantile: float = typer.Option(0.05, "--density-quantile", help="Quantile for low-density rejection."),
):
    """Apply abstention rules to a run."""
    run_dir = init_run_dir(outdir or run)
    get_logger(run_dir)
    update_manifest(run_dir, {"command": "trust.abstain", "inputs": [run], "tau": tau, "max_set_size": max_set_size})
    try:
        df = _load_predictions(run, label_col, proba_prefix)
        proba_cols = _select_proba_columns(df, proba_prefix)
        y_true = df[label_col].to_numpy(dtype=int)
        proba = df[proba_cols].to_numpy(dtype=float)

        density_scores = None
        if density_col and density_col in df.columns:
            density_scores = df[density_col].to_numpy(dtype=float)

        conformal_sets = None
        conformal_path = run_dir / "trust" / "conformal.json"
        if conformal_path.exists():
            conformal_payload = load_trust_payload(conformal_path)
            conformal_sets = conformal_payload.get("prediction_sets")

        result = apply_abstention_rules(
            proba,
            y_true,
            tau=tau,
            max_set_size=max_set_size,
            conformal_sets=conformal_sets,
            density_scores=density_scores,
            density_quantile=density_quantile,
        )
        density_threshold = None
        if density_scores is not None:
            density_threshold = float(np.quantile(density_scores, density_quantile))

        artifact = AbstentionArtifact(
            tau=tau,
            max_set_size=max_set_size,
            density_threshold=density_threshold,
            abstain_rate=result.abstain_rate,
            accuracy_on_answered=result.accuracy_on_answered,
            risk_coverage=result.risk_coverage,
            reasons=result.reasons,
        )
        trust_dir = _trust_dir(run_dir)
        abstain_path = trust_dir / "abstention.json"
        safe_json_dump(abstain_path, artifact.to_dict())

        trust_outputs = _update_trust_outputs(
            run_dir,
            {"abstention": artifact.to_dict(), "abstain_rate": artifact.abstain_rate},
        )
        readiness = evaluate_run_readiness(run, load_trust_payload(trust_outputs))
        readiness_artifact = ReadinessArtifact(
            score=readiness.score, components=readiness.components, notes=readiness.notes
        )
        readiness_path = trust_dir / "readiness.json"
        safe_json_dump(readiness_path, readiness_artifact.to_dict())
        _update_trust_outputs(run_dir, {"readiness": readiness_artifact.to_dict()})

        write_run_summary(
            run_dir,
            {
                "status": "success",
                "trust_outputs": str(trust_outputs),
                "artifacts": {
                    "abstention": str(abstain_path),
                    "readiness": str(readiness_path),
                    "trust_outputs": str(trust_outputs),
                },
            },
        )
        update_manifest(run_dir, {"artifacts": {"trust_outputs": str(trust_outputs)}})
        typer.echo("Abstention artifacts written.")
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@trust_app.command("calibrate-csv")
def calibrate_csv(
    calibration_csv: Path = typer.Option(..., "--calibration-csv", "-c", help="CSV with labels and probabilities."),
    label_col: str = typer.Option("y_true", help="Column containing true labels."),
    proba_prefix: str = typer.Option("proba_", help="Prefix for probability columns."),
    method: str = typer.Option("platt", help="Calibration method: platt|isotonic."),
    run_dir: Path = typer.Option(Path("runs/trust_calibration"), help="Run output directory."),
):
    """Legacy CSV calibration command."""
    run_dir = init_run_dir(run_dir)
    get_logger(run_dir)
    write_manifest(
        run_dir,
        {"command": "trust.calibrate-csv", "inputs": [calibration_csv], "method": method},
    )
    try:
        df = pd.read_csv(calibration_csv)
        if label_col not in df.columns:
            raise FoodSpecValidationError(f"label_col '{label_col}' not found in calibration CSV.")
        proba_cols = _select_proba_columns(df, proba_prefix)
        y_true = df[label_col].to_numpy(dtype=int)
        proba = df[proba_cols].to_numpy(dtype=float)
        metrics = compute_calibration_metrics(y_true, proba)
        output_path = run_dir / "calibration_metrics.json"
        safe_json_dump(output_path, {"method": method, "metrics": metrics})
        write_run_summary(run_dir, {"status": "success", "trust": "calibration", "metrics": str(output_path)})
        typer.echo("Calibration metrics written.")
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)


@trust_app.command("conformal-csv")
def conformal_csv(
    calibration_csv: Path = typer.Option(..., "--calibration-csv", "-c", help="CSV with labels and probabilities."),
    test_csv: Path = typer.Option(..., "--test-csv", "-t", help="CSV with probabilities (and optional labels)."),
    label_col: str = typer.Option("y_true", help="Column containing true labels."),
    proba_prefix: str = typer.Option("proba_", help="Prefix for probability columns."),
    alpha: float = typer.Option(0.1, help="Significance level (1 - target coverage)."),
    condition_col: Optional[str] = typer.Option(None, help="Optional column for Mondrian conditioning."),
    run_dir: Path = typer.Option(Path("runs/trust_conformal"), help="Run output directory."),
):
    """Legacy CSV conformal command."""
    run_dir = init_run_dir(run_dir)
    get_logger(run_dir)
    write_manifest(
        run_dir,
        {
            "command": "trust.conformal-csv",
            "inputs": [calibration_csv, test_csv],
            "alpha": alpha,
            "condition_col": condition_col,
        },
    )
    try:
        cal_df = pd.read_csv(calibration_csv)
        test_df = pd.read_csv(test_csv)
        if label_col not in cal_df.columns:
            raise FoodSpecValidationError(f"label_col '{label_col}' not found in calibration CSV.")
        proba_cols = _select_proba_columns(cal_df, proba_prefix)

        y_cal = cal_df[label_col].to_numpy()
        proba_cal = cal_df[proba_cols].to_numpy(dtype=float)
        meta_cal = cal_df[condition_col].to_numpy() if condition_col and condition_col in cal_df.columns else None

        cp = MondrianConformalClassifier(alpha=alpha, condition_key=condition_col)
        cp.fit(y_cal, proba_cal, meta_cal=meta_cal)

        if not set(proba_cols).issubset(test_df.columns):
            raise FoodSpecValidationError("Test CSV does not contain required probability columns.")
        proba_test = test_df[proba_cols].to_numpy(dtype=float)
        meta_test = test_df[condition_col].to_numpy() if condition_col and condition_col in test_df.columns else None
        y_true_test = test_df[label_col].to_numpy() if label_col in test_df.columns else None

        result = cp.predict_sets(proba_test, meta_test=meta_test, y_true=y_true_test)
        output_path = run_dir / "conformal_sets.csv"
        result.to_dataframe(y_true=y_true_test, bin_values=meta_test).to_csv(output_path, index=False)
        write_run_summary(run_dir, {"status": "success", "trust": "conformal", "sets": str(output_path)})
        typer.echo("Conformal prediction sets written.")
    except FoodSpecValidationError as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Validation error: {exc}", err=True)
        raise typer.Exit(code=2)
    except Exception as exc:
        write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(code=4)
