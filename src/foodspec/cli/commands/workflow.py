"""Workflow orchestration commands: experiment runner, benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer

from foodspec.apps.protocol_validation import run_protocol_benchmarks
from foodspec.config import load_config, merge_cli_overrides
from foodspec.core.api import FoodSpec
from foodspec.features.specs import FeatureSpec
from foodspec.logging_utils import get_logger, log_run_metadata
from foodspec.repro.experiment import ExperimentConfig, ExperimentEngine

logger = get_logger(__name__)

workflow_app = typer.Typer(help="Workflow orchestration commands")


def _apply_seeds(seeds: dict[str, Any]) -> None:
    """Set random seeds across common libraries if provided."""
    if not seeds:
        return
    if "python_random_seed" in seeds:
        import random as _random

        _random.seed(seeds["python_random_seed"])
    if "numpy_seed" in seeds:
        np.random.seed(int(seeds["numpy_seed"]))
    try:
        import torch  # type: ignore

        if "torch_seed" in seeds:
            torch.manual_seed(int(seeds["torch_seed"]))
    except Exception:
        pass


def _build_feature_specs(raw_specs: list[dict[str, Any]]) -> list[FeatureSpec]:
    """Construct FeatureSpec objects from YAML dictionaries."""
    specs: list[FeatureSpec] = []
    for spec in raw_specs:
        specs.append(
            FeatureSpec(
                name=spec.get("name", "feature"),
                ftype=spec.get("ftype", "band"),
                regions=spec.get("regions"),
                formula=spec.get("formula"),
                label=spec.get("label"),
                description=spec.get("description"),
                citation=spec.get("citation"),
                constraints=spec.get("constraints", {}),
                params=spec.get("params", {}),
            )
        )
    return specs


@workflow_app.command("run-exp")
def run_experiment(
    exp_path: Path = typer.Argument(..., help="Path to exp.yml experiment file."),
    output_dir: Optional[Path] = typer.Option(None, help="Override base output directory."),
    dry_run: bool = typer.Option(False, help="Only validate and summarize the experiment config."),
    artifact_path: Optional[Path] = typer.Option(None, help="Write a single-file .foodspec deployment artifact."),
):
    """Execute an experiment defined in exp.yml (reproducible, single-command pipeline)."""
    engine = ExperimentEngine.from_yaml(exp_path)
    cfg: ExperimentConfig = engine.config
    record = cfg.build_run_record()

    typer.echo(cfg.summary())
    typer.echo(f"config_hash={cfg.config_hash} dataset_hash={record.dataset_hash}")
    if dry_run:
        return

    _apply_seeds(cfg.seeds)

    # Load data and initialize workflow
    fs = FoodSpec(cfg.dataset.path, modality=cfg.dataset.modality)
    fs.bundle.run_record = record  # Attach provenance built from exp.yml

    label_column = (cfg.modeling.get("label_column") if cfg.modeling else None) or cfg.dataset.schema.get(
        "label_column", "label"
    )

    # QC
    if cfg.qc:
        qc_threshold = cfg.qc.get("threshold")
        if qc_threshold is None:
            qc_threshold = (cfg.qc.get("thresholds") or {}).get("outlier_rate", 0.5)
        fs.qc(method=cfg.qc.get("method", "robust_z"), threshold=qc_threshold)

    # Apply moats (optional, declarative)
    moats = cfg.moats or {}
    # 1) Matrix Correction (before preprocessing)
    mc = moats.get("matrix_correction")
    if mc:
        fs.apply_matrix_correction(
            method=mc.get("method", "adaptive_baseline"),
            scaling=mc.get("scaling", "median_mad"),
            domain_adapt=mc.get("domain_adapt", False),
            matrix_column=mc.get("matrix_column"),
        )

    # Preprocessing
    pre_cfg = cfg.preprocessing or {}
    pre_args = {k: v for k, v in pre_cfg.items() if k != "preset"}
    fs.preprocess(pre_cfg.get("preset", "auto"), **pre_args)

    # Features
    feat_cfg = cfg.features or {}
    feat_preset = feat_cfg.get("preset", "standard")
    feat_specs = _build_feature_specs(feat_cfg.get("specs", [])) if feat_preset == "specs" else None
    fs.features(feat_preset, specs=feat_specs)

    # 2) Heating Trajectory analysis (metrics only)
    ht = moats.get("heating_trajectory")
    if ht:
        fs.analyze_heating_trajectory(
            time_column=ht.get("time_column", "time_hours"),
            indices=ht.get("indices", ["pi", "tfc", "oit_proxy"]),
            classify_stages=ht.get("classify_stages", False),
            stage_column=ht.get("stage_column"),
            estimate_shelf_life=ht.get("estimate_shelf_life", False),
            shelf_life_threshold=ht.get("shelf_life_threshold", 2.0),
            shelf_life_index=ht.get("shelf_life_index", "pi"),
        )

    # 3) Calibration Transfer (before modeling)
    ct = moats.get("calibration_transfer")
    if ct:

        def _load_matrix(path):
            p = Path(path)
            if p.suffix == ".npy":
                return np.load(p)
            # assume CSV
            import pandas as pd

            return pd.read_csv(p, header=None).to_numpy()

        source_standards = _load_matrix(ct.get("source_standards")) if ct.get("source_standards") else None
        target_standards = _load_matrix(ct.get("target_standards")) if ct.get("target_standards") else None
        if source_standards is None or target_standards is None:
            raise typer.BadParameter("calibration_transfer requires source_standards and target_standards paths")
        fs.apply_calibration_transfer(
            source_standards=source_standards,
            target_standards=target_standards,
            method=ct.get("method", "ds"),
            pds_window_size=ct.get("pds_window_size", 11),
            alpha=ct.get("alpha", 1.0),
        )

    # Modeling
    mod_cfg = cfg.modeling or {}
    suite = mod_cfg.get("suite") or []
    first_model = suite[0] if suite else {}
    algorithm = first_model.get("algorithm", mod_cfg.get("algorithm", "rf"))
    cv_folds = first_model.get("cv_folds", mod_cfg.get("cv_folds", 5))
    if label_column not in fs.data.metadata.columns:
        raise typer.BadParameter(
            f"Label column '{label_column}' not found in metadata columns: {list(fs.data.metadata.columns)}"
        )
    params = first_model.get("params", {})
    fs.train(algorithm=algorithm, label_column=label_column, cv_folds=cv_folds, **params)

    # 4) Data Governance (metrics only)
    dg = moats.get("data_governance")
    if dg:
        # Derive label_column if not provided
        dg_label = dg.get("label_column") or label_column
        fs.summarize_dataset(label_column=dg_label)
        fs.check_class_balance(label_column=dg_label)
        if dg.get("replicate_column"):
            fs.assess_replicate_consistency(replicate_column=dg.get("replicate_column"))
        if dg.get("batch_column") or dg.get("replicate_column"):
            fs.detect_leakage(
                label_column=dg_label,
                batch_column=dg.get("batch_column"),
                replicate_column=dg.get("replicate_column"),
            )
        fs.compute_readiness_score(
            label_column=dg_label,
            batch_column=dg.get("batch_column"),
            replicate_column=dg.get("replicate_column"),
            required_metadata_columns=dg.get("required_metadata_columns"),
        )

    # Export
    base_dir = output_dir or cfg.outputs.get("base_dir") or Path("foodspec_runs") / record.run_id
    out_path = fs.export(base_dir)
    if artifact_path is None:
        artifact_path = Path(base_dir) / f"{record.run_id}.foodspec"
    try:
        from foodspec.artifact import save_artifact

        target_grid = getattr(fs.data, "wavenumbers", None)
        save_artifact(
            fs.bundle,
            artifact_path,
            target_grid=target_grid,
            feature_specs=feat_specs,
            schema={"label_column": label_column, "modality": cfg.dataset.modality},
        )
        typer.echo(f"Artifact saved: {artifact_path}")
    except Exception as exc:  # pragma: no cover - defensive
        typer.echo(f"Warning: failed to write artifact: {exc}", err=True)
    typer.echo(f"Experiment complete. Outputs: {out_path}")


@workflow_app.command("protocol-benchmarks")
def protocol_benchmarks(
    output_dir: str = typer.Option("./protocol_benchmarks", help="Directory to write benchmark metrics."),
    random_state: int = typer.Option(42, help="Random seed."),
    config: Optional[str] = typer.Option(None, "--config", help="Optional YAML/JSON config."),
):
    """Run protocol benchmarks on public datasets and save reports."""
    base_cfg = {"output_dir": output_dir, "random_state": random_state}
    cfg = load_config(config) if config else base_cfg
    cfg = merge_cli_overrides(cfg, base_cfg)

    out_path = Path(cfg["output_dir"])
    run_meta = log_run_metadata(logger, {"command": "protocol-benchmarks"})
    summary = run_protocol_benchmarks(out_path, random_state=cfg.get("random_state", random_state))
    # write run metadata alongside metrics
    meta_path = out_path / "run_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    typer.echo("Protocol benchmarks summary:")
    typer.echo(json.dumps(summary, indent=2))


@workflow_app.command("bench")
def bench():
    """Run protocol benchmarks (alias for protocol-benchmarks)."""
    run_protocol_benchmarks(Path("./protocol_benchmarks"))
