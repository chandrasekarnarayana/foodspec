"""Phase 1: Minimal orchestrator for guaranteed E2E workflow.

This is the minimal orchestrator that guarantees:
- Sequential execution of pipeline stages
- Proper error handling with exit codes
- Manifest + artifact tracking
- Logging (human-readable + structured JSON)

Entry point: run_workflow(cfg: WorkflowConfig)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from foodspec.modeling.api import fit_predict

from .artifact_contract import (
    ArtifactContract,
    write_success_json,
)
from .config import WorkflowConfig
from .errors import (
    EXIT_SUCCESS,
    ArtifactError,
    ModelingError,
    ProtocolError,
    ValidationError,
    WorkflowError,
    classify_error_type,
    write_error_json,
)
from .fingerprint import Manifest, compute_dataset_fingerprint
from .qc_gates import (
    DataIntegrityGate,
    GateResult,
    SpectralQualityGate,
)
from .model_registry import resolve_model_name
from .regulatory import (
    enforce_model_approved,
    enforce_reporting,
    enforce_trust_stack,
)

logger = logging.getLogger(__name__)


def _setup_run_dir(output_dir: Optional[Path | str]) -> Path:
    """Initialize run directory with subdirectories."""
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"run_{stamp}"
    else:
        run_dir = Path(output_dir)

    # Create subdirectories
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    return run_dir


def _setup_logging(run_dir: Path, verbose: bool = False) -> Tuple[logging.Logger, Any]:
    """Setup logging to run.log + run.jsonl.
    
    Returns
    -------
    (logger, jsonl_handler)
        Logger and reference to JSON file handle.
    """
    log_file = run_dir / "logs" / "run.log"
    jsonl_file = run_dir / "logs" / "run.jsonl"

    # Configure logger
    logger = logging.getLogger("foodspec.workflow")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler (human-readable)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # JSON lines file handler (structured)
    jsonl_fh = logging.FileHandler(jsonl_file)
    jsonl_fh.setLevel(logging.DEBUG if verbose else logging.INFO)

    class JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "stage": record.name.replace("foodspec.workflow.", ""),
                "message": record.getMessage(),
            }
            return json.dumps(data)

    jsonl_fh.setFormatter(JSONFormatter())
    logger.addHandler(jsonl_fh)

    logger.info("=== Workflow started ===")
    return logger, (log_file, jsonl_file)


def _validate_inputs(cfg: WorkflowConfig) -> Tuple[bool, list[str]]:
    """Validate workflow config.
    
    Returns
    -------
    (is_valid, errors)
    """
    is_valid, errors = cfg.validate()
    if not is_valid:
        for error in errors:
            logger.error(error)
    return is_valid, errors


def _load_and_validate_protocol(
    protocol_path: Path,
) -> Tuple[Any, Dict[str, Any]]:
    """Load protocol and validate against schema.
    
    Returns
    -------
    (protocol_config, validation_results)
    """
    logger.info(f"Loading protocol: {protocol_path}")

    try:
        # Use ProtocolConfig directly instead of load_protocol which searches in examples/protocols
        from foodspec.protocol.config import ProtocolConfig
        protocol_cfg = ProtocolConfig.from_file(protocol_path)
        logger.info(f"Protocol loaded: version={getattr(protocol_cfg, 'version', 'unknown')}")
        return protocol_cfg, {"status": "valid"}
    except Exception as e:
        logger.error(f"Protocol load failed: {e}")
        raise ProtocolError(
            message=f"Failed to load protocol: {e}",
            stage="protocol_loading",
            hint="Check protocol YAML syntax and required fields.",
        ) from e


def _read_and_validate_data(
    input_paths: list[Path],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read CSV and validate basic schema.
    
    Returns
    -------
    (dataframe, validation_results)
    """
    if not input_paths:
        raise ValidationError(
            message="No input files provided",
            stage="data_loading",
            hint="Specify at least one --input CSV file.",
        )

    logger.info(f"Loading {len(input_paths)} input file(s)...")

    try:
        # Phase 1: use first input
        df = pd.read_csv(input_paths[0])
        logger.info(
            f"Loaded CSV: {input_paths[0].name} "
            f"({len(df)} rows, {len(df.columns)} cols)"
        )

        # Compute fingerprint
        fp = compute_dataset_fingerprint(input_paths[0])
        logger.info(f"Dataset SHA256: {fp.get('sha256', 'N/A')[:8]}...")

        validation_result = {
            "status": "valid",
            "rows": len(df),
            "columns": list(df.columns),
            "fingerprint": fp,
        }
        return df, validation_result

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise ValidationError(
            message=f"Input file not found: {e}",
            stage="data_loading",
            hint="Check file path and ensure it exists.",
        ) from e
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise ValidationError(
            message=f"Failed to read CSV: {e}",
            stage="data_loading",
            hint="Ensure CSV is well-formed and readable.",
        ) from e


def _run_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run preprocessing stage (stub for Phase 1).
    
    Phase 1: minimal preprocessing. Just return df as-is.
    
    Returns
    -------
    (preprocessed_df, stage_result)
    """
    logger.info("Preprocessing stage: running...")
    logger.info(f"Preprocessing: input shape {df.shape}")
    # TODO: integrate actual preprocessing pipeline
    return df, {"status": "skipped", "reason": "Phase 1 stub"}


def _run_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run feature extraction stage (stub for Phase 1).
    
    Returns
    -------
    (features_df, stage_result)
    """
    logger.info("Feature extraction stage: running...")
    logger.info(f"Features: output shape {df.shape}")
    # TODO: integrate actual feature extraction
    return df, {"status": "skipped", "reason": "Phase 1 stub"}


def _run_qc_gates(
    df: pd.DataFrame,
    enforce: bool = False,
    label_col: Optional[str] = None,
) -> Tuple[bool, Dict[str, GateResult]]:
    """Run QC gates and return pass/fail status.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    enforce : bool
        If True, fail on gate failures. If False, only warn.
    label_col : Optional[str]
        Label column for class balance check.
    
    Returns
    -------
    (passed, gate_results)
        (all_gates_passed, {gate_name: GateResult})
    """
    logger.info(f"QC gates: running (enforce={enforce})...")

    gate_results = {}
    all_passed = True

    # Data integrity gate
    di_gate = DataIntegrityGate()
    di_result = di_gate.run(df, label_col=label_col)
    gate_results["data_integrity"] = di_result
    logger.info(f"  Data Integrity: {di_result.status}")
    if di_result.status == "fail":
        all_passed = False

    # Spectral quality gate
    spectral_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col and label_col in spectral_cols:
        spectral_cols.remove(label_col)

    sq_gate = SpectralQualityGate()
    sq_result = sq_gate.run(df, spectral_cols=spectral_cols if spectral_cols else None)
    gate_results["spectral_quality"] = sq_result
    logger.info(f"  Spectral Quality: {sq_result.status}")
    if sq_result.status == "fail":
        all_passed = False

    # Log gate details
    for gate_name, gate_result in gate_results.items():
        logger.debug(f"    {gate_name}: {gate_result.message}")
        for metric_name, metric_val in gate_result.metrics.items():
            logger.debug(f"      {metric_name}: {metric_val}")

    # Decide action
    if not all_passed:
        if enforce:
            logger.error("QC gates failed with enforce=True")
        else:
            logger.warning("QC gates failed but enforce=False (advisory only)")
    else:
        logger.info("✅ All QC gates passed")

    return all_passed, gate_results


def _run_modeling(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: Optional[str] = None,
    scheme: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run modeling stage.
    
    Returns
    -------
    stage_result dict with metrics, model info, etc.
    """
    logger.info("Modeling stage: running...")

    try:
        if model_name is None:
            model_name = "logreg"
        if scheme is None:
            scheme = "random"

        model_name = resolve_model_name(model_name) or model_name
        logger.info(f"Model: {model_name} | Scheme: {scheme}")

        # Convert to numpy for fit_predict
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else y

        result = fit_predict(
            X_arr,
            y_arr,
            model_name=model_name,
            scheme=scheme,
            seed=seed or 0,
            allow_random_cv=True,  # Allow random CV for testing/research
        )

        logger.info(f"Modeling complete: accuracy={result.metrics.get('accuracy', 'N/A')}")

        return {
            "status": "success",
            "model": model_name,
            "scheme": scheme,
            "metrics": result.metrics,
            "folds": result.folds if hasattr(result, "folds") else [],
        }
    except Exception as e:
        logger.error(f"Modeling failed: {e}")
        raise ModelingError(
            message=f"Modeling failed: {e}",
            stage="modeling",
            hint="Check feature matrix shape and label encoding.",
        ) from e


def run_workflow(cfg: WorkflowConfig) -> int:
    """Execute minimal guaranteed workflow.
    
    Orchestrates:
    1. Config validation
    2. Protocol loading
    3. Data loading + fingerprinting
    4. Preprocessing (stub)
    5. Feature extraction (stub)
    6. Modeling (if requested)
    7. Trust stack (stub)
    8. Reporting (stub)
    9. Artifact validation
    
    Parameters
    ----------
    cfg : WorkflowConfig
        Workflow configuration.
    
    Returns
    -------
    int
        Exit code (0 for success, 2-9 for errors).
    
    Examples
    --------
    Execute a minimal workflow::
    
        cfg = WorkflowConfig(
            protocol=Path("protocol.yaml"),
            inputs=[Path("data.csv")],
            output_dir=Path("runs/exp1"),
            mode="research",
            seed=42,
        )
        exit_code = run_workflow(cfg)
    """
    # Setup run directory
    run_dir = _setup_run_dir(cfg.output_dir)
    logger_ref, log_files = _setup_logging(run_dir, cfg.verbose)

    try:
        logger_ref.info(f"Run directory: {run_dir}")
        logger_ref.info(cfg.summary())

        # Validate config
        is_valid, errors = _validate_inputs(cfg)
        if not is_valid:
            raise ValidationError(
                message=f"Config validation failed: {errors}",
                stage="config_validation",
                hint="Check CLI arguments and protocol path.",
            )

        # Load protocol
        protocol_cfg, proto_result = _load_and_validate_protocol(cfg.protocol)
        logger_ref.info(f"Protocol validation: {proto_result}")

        # Load data
        df, data_result = _read_and_validate_data(cfg.inputs)
        logger_ref.info(f"Data validation: {len(df)} rows")

        # ==== Phase 2: QC Gates ====
        # Infer label column (needed for QC)
        label_col = cfg.label_col or "label"
        if label_col not in df.columns:
            possible_labels = [c for c in df.columns if "label" in c.lower()]
            if possible_labels:
                label_col = possible_labels[0]
                logger_ref.info(f"Inferred label column: {label_col}")
            else:
                label_col = None

        # Run QC gates
        qc_results = {}
        if cfg.enforce_qc:
            qc_passed, qc_results = _run_qc_gates(
                df,
                enforce=True,
                label_col=label_col,
            )

            # Fail on gate failure when explicitly requested
            if not qc_passed:
                failed_gates = [name for name, res in qc_results.items() if res.status == "fail"]
                raise ArtifactError(
                    message=f"QC gates failed: {', '.join(failed_gates)}",
                    stage="qc_gates",
                    hint="Review QC metrics in artifacts/qc_results.json and adjust data or thresholds.",
                )

        # ==== Phase 2: Regulatory Enforcement ====
        if cfg.mode == "regulatory":
            # Check model approval
            model_name = cfg.model or "LogisticRegression"
            is_approved, approval_msg = enforce_model_approved(model_name, cfg.mode)
            logger_ref.info(f"Model approval: {approval_msg}")
            if not is_approved:
                raise ArtifactError(
                    message=f"Model not approved for regulatory mode: {model_name}",
                    stage="regulatory_enforcement",
                    hint=approval_msg,
                )

            # Check trust stack requirement (only if explicitly requested via CLI flag)
            if cfg.enable_trust:
                trust_ok, trust_msg = enforce_trust_stack(cfg.enable_trust, cfg.mode)
                logger_ref.info(f"Trust stack: {trust_msg}")
                if not trust_ok:
                    raise ArtifactError(
                        message=f"Trust stack requirement failed: {trust_msg}",
                        stage="regulatory_enforcement",
                        hint="Enable trust stack with --enable-trust",
                    )

            # Check reporting requirement (only if explicitly requested via CLI flag)
            if cfg.enable_reporting:
                report_ok, report_msg = enforce_reporting(cfg.enable_reporting, cfg.mode)
                logger_ref.info(f"Reporting: {report_msg}")
                if not report_ok:
                    raise ArtifactError(
                        message=f"Reporting requirement failed: {report_msg}",
                        stage="regulatory_enforcement",
                        hint="Enable reporting with --enable-report",
                    )

        # Preprocessing (stub)
        if cfg.enable_preprocessing:
            df, preproc_result = _run_preprocessing(df)
            logger_ref.info(f"Preprocessing: {preproc_result}")

        # Feature extraction (stub)
        if cfg.enable_features:
            X_features, features_result = _run_features(df)
            logger_ref.info(f"Features: {features_result}")
        else:
            X_features = df

        # Modeling
        modeling_result = {"status": "skipped"}
        if cfg.enable_modeling and len(X_features) > 1:
            # Infer label and group columns
            label_col = cfg.label_col or "label"
            if label_col not in X_features.columns:
                # Try to find label column
                possible_labels = [c for c in X_features.columns if "label" in c.lower()]
                if possible_labels:
                    label_col = possible_labels[0]
                    logger_ref.info(f"Inferred label column: {label_col}")
                else:
                    logger_ref.warning("No label column found; skipping modeling")
                    label_col = None

            if label_col and label_col in X_features.columns:
                y = X_features[label_col]
                X = X_features.drop(columns=[label_col])
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    logger_ref.warning("No numeric feature columns found; skipping modeling")
                else:
                    dropped = [c for c in X.columns if c not in numeric_cols]
                    if dropped:
                        logger_ref.warning(
                            f"Dropping non-numeric feature columns before modeling: {dropped}"
                        )
                    X = X[numeric_cols]
                    modeling_result = _run_modeling(
                        X,
                        y,
                        model_name=cfg.model,
                        scheme=cfg.scheme,
                        seed=cfg.seed,
                    )

        # Build manifest
        logger_ref.info("Building manifest...")
        manifest = Manifest.build(
            protocol_path=cfg.protocol,
            input_paths=cfg.inputs,
            seed=cfg.seed,
            mode=cfg.mode,
            cli_args={
                "mode": cfg.mode,
                "seed": cfg.seed,
                "scheme": cfg.scheme,
                "model": cfg.model,
            },
            cli_overrides=cfg.cli_overrides,
        )
        manifest.finalize()
        manifest.artifacts = {
            "data_fingerprint": data_result.get("fingerprint", {}),
            "modeling": modeling_result,
            "qc": {name: res.to_dict() for name, res in qc_results.items()},
        }
        manifest_path = run_dir / "manifest.json"
        manifest.save(manifest_path)
        logger_ref.info(f"Manifest saved: {manifest_path}")

        # Save QC results to artifacts
        if qc_results:
            qc_file = run_dir / "artifacts" / "qc_results.json"
            qc_file.parent.mkdir(parents=True, exist_ok=True)
            with open(qc_file, "w") as f:
                json.dump(
                    {name: res.to_dict() for name, res in qc_results.items()},
                    f,
                    indent=2,
                )
            logger_ref.info(f"QC results saved: {qc_file}")

        # Validate artifact contract (success)
        is_valid, missing = ArtifactContract.validate_success(
            run_dir,
            enforce_qc=cfg.enforce_qc,
            enable_trust=cfg.enable_trust,
            enable_reporting=cfg.enable_reporting,
        )
        if not is_valid:
            logger_ref.warning(f"Artifact contract incomplete: missing {missing}")
            # Phase 1: warn but don't fail
        else:
            logger_ref.info("✅ Artifact contract validated")

        # Write success marker
        success_summary = {
            "protocol": str(cfg.protocol),
            "inputs": [str(p) for p in cfg.inputs],
            "mode": cfg.mode,
            "seed": cfg.seed,
            "rows": len(df),
            "columns": len(df.columns),
            "modeling": modeling_result,
        }
        write_success_json(run_dir, success_summary)

        logger_ref.info("=== Workflow completed successfully ===")
        return EXIT_SUCCESS

    except WorkflowError as exc:
        # Structured error handling
        logger_ref.error(f"Workflow error [{exc.stage}]: {exc.message}")
        if exc.hint:
            logger_ref.info(f"Hint: {exc.hint}")

        # Write error.json
        error_path = write_error_json(
            run_dir,
            exc,
            stage=exc.stage,
            exit_code=exc.exit_code,
            hint=exc.hint,
        )
        logger_ref.info(f"Error details saved: {error_path}")

        # Validate artifact contract (failure)
        is_valid, missing = ArtifactContract.validate_failure(run_dir)
        if not is_valid:
            logger_ref.warning(f"Artifact contract incomplete on failure: missing {missing}")

        logger_ref.error(f"=== Workflow failed (exit {exc.exit_code}) ===")
        return exc.exit_code

    except Exception as exc:
        # Classify and convert to WorkflowError
        logger_ref.error(f"Unexpected error: {exc}", exc_info=True)
        workflow_error, exit_code = classify_error_type(exc)
        workflow_error.stage = "unknown"

        error_path = write_error_json(
            run_dir,
            workflow_error,
            stage=workflow_error.stage,
            exit_code=exit_code,
        )
        logger_ref.info(f"Error details saved: {error_path}")

        logger_ref.error(f"=== Workflow failed (exit {exit_code}) ===")
        return exit_code
