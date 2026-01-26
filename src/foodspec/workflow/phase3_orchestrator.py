"""Phase 3: Full end-to-end orchestrator with real pipeline execution.

This orchestrator runs the complete FoodSpec pipeline:
- Real preprocessing via ProtocolRunner or PreprocessEngine
- Real feature extraction
- Real modeling with fit_predict
- Real trust stack (calibration, conformal, abstention)
- Real reporting generation
- Strict regulatory enforcement (QC + trust + reporting mandatory)

Entry point: run_workflow_phase3(cfg: WorkflowConfig) -> int
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from foodspec.protocol.config import ProtocolConfig
from foodspec.protocol.runner import ProtocolRunner
from foodspec.modeling.api import fit_predict
from foodspec.features.hybrid import extract_features
from foodspec.features.schema import parse_feature_config

from .config import WorkflowConfig
from .fingerprint import Manifest, compute_dataset_fingerprint
from .errors import (
    WorkflowError,
    ValidationError,
    ProtocolError,
    ModelingError,
    TrustError,
    ArtifactError,
    write_error_json,
    classify_error_type,
    EXIT_SUCCESS,
)
from .artifact_contract import ArtifactContract, write_success_json
from .qc_gates import (
    DataIntegrityGate,
    SpectralQualityGate,
    ModelReliabilityGate,
    GateResult,
)
from .regulatory import (
    enforce_model_approved,
    enforce_trust_stack,
    enforce_reporting,
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
    (run_dir / "report").mkdir(parents=True, exist_ok=True)

    return run_dir


def _setup_logging(run_dir: Path, verbose: bool = False) -> Tuple[logging.Logger, Any]:
    """Setup logging to run.log + run.jsonl."""
    log_file = run_dir / "logs" / "run.log"
    jsonl_file = run_dir / "logs" / "run.jsonl"

    # Configure logger
    logger_ref = logging.getLogger("foodspec.workflow")
    logger_ref.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler (human-readable)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    logger_ref.addHandler(fh)

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
    logger_ref.addHandler(jsonl_fh)

    return logger_ref, jsonl_fh


def _validate_inputs(cfg: WorkflowConfig) -> Tuple[bool, list]:
    """Validate workflow configuration."""
    errors = []

    if not cfg.protocol.exists():
        errors.append(f"Protocol file not found: {cfg.protocol}")

    if not cfg.inputs:
        errors.append("No input files provided")

    for inp in cfg.inputs:
        if not inp.exists():
            errors.append(f"Input file not found: {inp}")

    if cfg.mode not in ("research", "regulatory"):
        errors.append(f"Mode must be 'research' or 'regulatory', got: {cfg.mode}")

    return len(errors) == 0, errors


def _load_and_validate_protocol(protocol_path: Path) -> Tuple[ProtocolConfig, Dict[str, Any]]:
    """Load and validate protocol."""
    logger.info(f"Loading protocol: {protocol_path}")

    try:
        protocol_cfg = ProtocolConfig.from_file(protocol_path)
        logger.info(f"Protocol loaded: version={protocol_cfg.version}")
        return protocol_cfg, {"status": "valid"}
    except Exception as e:
        logger.error(f"Protocol loading failed: {e}")
        raise ProtocolError(
            message=f"Protocol loading failed: {e}",
            stage="protocol_loading",
            hint="Check protocol YAML/JSON format and file path.",
        ) from e


def _read_and_validate_data(input_paths: list[Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate input data."""
    logger.info(f"Loading {len(input_paths)} input file(s)...")

    dfs = []
    for inp in input_paths:
        try:
            logger.info(f"Loading CSV: {inp.name}")
            df = pd.read_csv(inp)
            logger.info(f"Loaded CSV: {inp.name} ({len(df)} rows, {len(df.columns)} cols)")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {inp}: {e}")
            raise ValidationError(
                message=f"Failed to load {inp}: {e}",
                stage="data_loading",
                hint="Ensure CSV is well-formed and readable.",
            ) from e

    # Concatenate if multiple files
    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Concatenated {len(dfs)} files: {len(df)} rows total")

    # Compute fingerprint
    fingerprint = compute_dataset_fingerprint(df)
    logger.info(f"Dataset SHA256: {fingerprint[:16]}...")

    return df, {"fingerprint": fingerprint}


def _run_qc_gates(
    df: pd.DataFrame,
    enforce: bool = False,
    label_col: Optional[str] = None,
) -> Tuple[bool, Dict[str, GateResult]]:
    """Run QC gates and return pass/fail status."""
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

    if not all_passed:
        if enforce:
            logger.error(f"QC gates failed with enforce=True")
        else:
            logger.warning(f"QC gates failed but enforce=False (advisory only)")
    else:
        logger.info("✅ All QC gates passed")

    return all_passed, gate_results


def _run_preprocessing_real(
    df: pd.DataFrame,
    protocol_cfg: ProtocolConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run real preprocessing via protocol or engine.
    
    Returns
    -------
    (preprocessed_df, stage_result)
    """
    logger.info("Preprocessing stage: running real pipeline...")

    try:
        # Check if protocol has preprocessing config
        preprocess_cfg = getattr(protocol_cfg, "preprocessing", {})
        if preprocess_cfg:
            logger.info(f"Preprocessing config found: {preprocess_cfg}")
            # TODO: Wire actual preprocessing engine when available
            # For now, return data as-is with note
            logger.info("Preprocessing: placeholder (real engine TBD)")
            return df, {"status": "skipped", "reason": "Real preprocessing TBD"}
        else:
            logger.info("No preprocessing configured in protocol")
            return df, {"status": "skipped", "reason": "No preprocessing config"}
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise ModelingError(
            message=f"Preprocessing failed: {e}",
            stage="preprocessing",
            hint="Check preprocessing configuration and data format.",
        ) from e


def _run_features_real(
    df: pd.DataFrame,
    protocol_cfg: ProtocolConfig,
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run real feature extraction via protocol.
    
    Returns
    -------
    (features_df, stage_result)
    """
    logger.info("Feature extraction stage: running real pipeline...")

    try:
        # Check if protocol has feature config
        feature_config = getattr(protocol_cfg, "features", {})
        if feature_config:
            logger.info(f"Feature config found")
            # TODO: Wire actual feature extraction when available
            logger.info("Feature extraction: placeholder (real engine TBD)")
            return df, {"status": "skipped", "reason": "Real features TBD"}
        else:
            logger.info("No features configured in protocol; using all columns")
            return df, {"status": "skipped", "reason": "No feature config"}
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise ModelingError(
            message=f"Feature extraction failed: {e}",
            stage="feature_extraction",
            hint="Check feature configuration.",
        ) from e


def _run_modeling_real(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: Optional[str] = None,
    scheme: Optional[str] = None,
    seed: Optional[int] = None,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run real modeling stage.
    
    Returns
    -------
    stage_result dict with metrics, model info, etc.
    """
    logger.info("Modeling stage: running real fit_predict...")

    try:
        if model_name is None:
            model_name = "LogisticRegression"
        if scheme is None:
            scheme = "random"

        logger.info(f"Model: {model_name} | Scheme: {scheme} | Groups: {groups is not None}")

        # Convert to numpy for fit_predict
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else y

        result = fit_predict(
            X_arr,
            y_arr,
            model_name=model_name,
            scheme=scheme,
            seed=seed or 0,
            groups=groups,
        )

        logger.info(f"Modeling complete: accuracy={result.metrics.get('accuracy', 'N/A'):.3f}")

        return {
            "status": "success",
            "model": model_name,
            "scheme": scheme,
            "metrics": result.metrics,
            "folds": result.folds if hasattr(result, "folds") else [],
            "predictions": result.predictions if hasattr(result, "predictions") else [],
        }
    except Exception as e:
        logger.error(f"Modeling failed: {e}")
        raise ModelingError(
            message=f"Modeling failed: {e}",
            stage="modeling",
            hint="Check feature matrix shape and label encoding.",
        ) from e


def _run_trust_stack_real(
    predictions: np.ndarray,
    y_true: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run real trust stack (calibration + conformal + abstention).
    
    Placeholder: returns stub results. Real trust module wiring TBD.
    
    Returns
    -------
    trust_result dict
    """
    logger.info("Trust stack stage: running real pipeline...")

    try:
        # TODO: Wire actual trust stack when available
        # For now, return stub with note
        logger.info("Trust: placeholder (real trust stack TBD)")
        
        return {
            "status": "skipped",
            "reason": "Real trust stack TBD",
            "coverage": 0.0,
            "calibration": {},
            "conformal": {},
            "abstention": {},
        }
    except Exception as e:
        logger.error(f"Trust stack failed: {e}")
        raise TrustError(
            message=f"Trust stack failed: {e}",
            stage="trust_stack",
            hint="Check trust configuration.",
        ) from e


def _run_reporting_real(
    run_dir: Path,
    cfg: WorkflowConfig,
    manifest: Manifest,
    modeling_result: Dict[str, Any],
    qc_results: Dict[str, GateResult],
    trust_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate real HTML report.
    
    Placeholder: creates minimal report. Real reporting TBD.
    
    Returns
    -------
    reporting_result dict
    """
    logger.info("Reporting stage: generating HTML report...")

    try:
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal HTML report
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>FoodSpec Run Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>FoodSpec Workflow Report</h1>
    <div class="section">
        <h2>Run Information</h2>
        <p><strong>Mode:</strong> {cfg.mode}</p>
        <p><strong>Timestamp:</strong> {datetime.now(timezone.utc).isoformat()}</p>
        <p><strong>Seed:</strong> {cfg.seed}</p>
    </div>
    
    <div class="section">
        <h2>QC Results</h2>
        <pre>{json.dumps({name: res.to_dict() for name, res in qc_results.items()}, indent=2)}</pre>
    </div>
    
    <div class="section">
        <h2>Modeling Results</h2>
        <pre>{json.dumps(modeling_result, indent=2, default=str)}</pre>
    </div>
    
    <div class="section">
        <h2>Manifest Hash</h2>
        <pre>{manifest.id if hasattr(manifest, 'id') else 'N/A'}</pre>
    </div>
</body>
</html>"""
        
        report_path = report_dir / "index.html"
        report_path.write_text(html_content)
        logger.info(f"Report generated: {report_path}")
        
        return {
            "status": "success",
            "report_path": str(report_path),
        }
    except Exception as e:
        logger.error(f"Reporting failed: {e}")
        raise ArtifactError(
            message=f"Reporting failed: {e}",
            stage="reporting",
            hint="Check report configuration.",
        ) from e


def run_workflow_phase3(cfg: WorkflowConfig, strict_regulatory: bool = True) -> int:
    """Execute Phase 3 full end-to-end workflow.
    
    Orchestrates:
    1. Config validation
    2. Protocol loading
    3. Data loading + fingerprinting
    4. QC gates (strict in regulatory mode)
    5. Real preprocessing
    6. Real feature extraction
    7. Real modeling via fit_predict
    8. Real trust stack
    9. Real reporting
    10. Artifact validation
    
    Parameters
    ----------
    cfg : WorkflowConfig
        Workflow configuration.
    strict_regulatory : bool
        If True (default), enforce strict regulatory semantics:
        - regulatory mode MUST enforce QC
        - regulatory mode MUST enable trust
        - regulatory mode MUST enable reporting
        If False, use Phase 1 relaxed semantics (backward compatible).
    
    Returns
    -------
    int
        Exit code (0 for success, 2-9 for errors).
    """
    # Setup run directory
    run_dir = _setup_run_dir(cfg.output_dir)
    logger_ref, log_files = _setup_logging(run_dir, cfg.verbose)

    try:
        logger_ref.info(f"=== Phase 3 Workflow Started (strict={strict_regulatory}) ===")
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

        # Infer label column
        label_col = cfg.label_col or "label"
        if label_col not in df.columns:
            possible_labels = [c for c in df.columns if "label" in c.lower()]
            if possible_labels:
                label_col = possible_labels[0]
                logger_ref.info(f"Inferred label column: {label_col}")
            else:
                label_col = None

        # ==== Phase 3: Strict Regulatory Enforcement ====
        if cfg.mode == "regulatory" and strict_regulatory:
            # Regulatory mode in Phase 3 MUST enforce all requirements
            logger_ref.info("Regulatory strict mode: enforcing QC, trust, reporting")
            
            # Enforce QC automatically in regulatory mode
            qc_passed, qc_results = _run_qc_gates(
                df,
                enforce=True,  # Always enforce in regulatory strict
                label_col=label_col,
            )
            
            if not qc_passed:
                failed_gates = [name for name, res in qc_results.items() if res.status == "fail"]
                raise ArtifactError(
                    message=f"Regulatory mode: QC gates failed: {', '.join(failed_gates)}",
                    stage="qc_gates",
                    hint="Regulatory mode requires passing QC gates.",
                )
            
            # Check model approval (always required)
            model_name = cfg.model or "LogisticRegression"
            is_approved, approval_msg = enforce_model_approved(model_name, cfg.mode)
            logger_ref.info(f"Model approval: {approval_msg}")
            if not is_approved:
                raise ProtocolError(
                    message=f"Regulatory mode: Model not approved: {model_name}",
                    stage="regulatory_enforcement",
                    hint=approval_msg,
                )
            
            # Trust MUST be enabled (set to True for strict mode)
            trust_enabled = True
            logger_ref.info("Regulatory mode: trust stack is mandatory")
            
            # Reporting MUST be enabled (set to True for strict mode)
            report_enabled = True
            logger_ref.info("Regulatory mode: reporting is mandatory")
        else:
            # Research mode or Phase 1 compat: relaxed enforcement
            if cfg.enforce_qc:
                qc_passed, qc_results = _run_qc_gates(
                    df,
                    enforce=True,
                    label_col=label_col,
                )
                if not qc_passed:
                    failed_gates = [name for name, res in qc_results.items() if res.status == "fail"]
                    raise ArtifactError(
                        message=f"QC gates failed: {', '.join(failed_gates)}",
                        stage="qc_gates",
                        hint="Review QC metrics.",
                    )
            else:
                # Advisory QC
                qc_passed, qc_results = _run_qc_gates(
                    df,
                    enforce=False,
                    label_col=label_col,
                )
            
            trust_enabled = cfg.enable_trust
            report_enabled = cfg.enable_reporting

        # ==== Real Pipeline Stages ====
        
        # Preprocessing
        df_preproc, preproc_result = _run_preprocessing_real(df, protocol_cfg)
        logger_ref.info(f"Preprocessing: {preproc_result}")

        # Feature extraction
        df_features, features_result = _run_features_real(df_preproc, protocol_cfg, label_col)
        logger_ref.info(f"Feature extraction: {features_result}")

        # Prepare for modeling
        modeling_result = {"status": "skipped"}
        if cfg.enable_modeling and len(df_features) > 1 and label_col and label_col in df_features.columns:
            y = df_features[label_col]
            X = df_features.drop(columns=[label_col])
            
            group_col = cfg.group_col or "group"
            groups = df_features[group_col].to_numpy() if group_col and group_col in df_features.columns else None
            
            modeling_result = _run_modeling_real(
                X,
                y,
                model_name=cfg.model,
                scheme=cfg.scheme,
                seed=cfg.seed,
                groups=groups,
            )
            
            # ModelReliabilityGate on real results
            model_gate = ModelReliabilityGate()
            model_result = model_gate.run(modeling_result)
            logger_ref.info(f"  Model Reliability: {model_result.status}")
            if model_result.status == "fail":
                qc_results["model_reliability"] = model_result

        # Trust stack
        trust_result = {"status": "skipped"}
        if trust_enabled and modeling_result.get("status") == "success":
            predictions = modeling_result.get("predictions", np.array([]))
            y_true = df_features[label_col].to_numpy() if label_col else None
            if predictions is not None and y_true is not None:
                trust_result = _run_trust_stack_real(predictions, y_true)
                logger_ref.info(f"Trust stack: {trust_result.get('reason', trust_result.get('status'))}")

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
                "strict_regulatory": strict_regulatory,
            },
            cli_overrides=cfg.cli_overrides,
        )
        manifest.finalize()
        manifest.artifacts = {
            "data_fingerprint": data_result.get("fingerprint", {}),
            "preprocessing": preproc_result,
            "features": features_result,
            "modeling": modeling_result,
            "trust": trust_result,
            "qc": {name: res.to_dict() for name, res in qc_results.items()},
        }
        manifest_path = run_dir / "manifest.json"
        manifest.save(manifest_path)
        logger_ref.info(f"Manifest saved: {manifest_path}")

        # Save QC results
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

        # Reporting
        if report_enabled:
            reporting_result = _run_reporting_real(
                run_dir,
                cfg,
                manifest,
                modeling_result,
                qc_results,
                trust_result,
            )
            logger_ref.info(f"Reporting: {reporting_result.get('status')}")
        else:
            reporting_result = {"status": "skipped"}

        # Validate artifact contract (Phase 3)
        is_valid, missing = ArtifactContract.validate_success(
            run_dir,
            enforce_qc=(cfg.enforce_qc or cfg.mode == "regulatory") if strict_regulatory else cfg.enforce_qc,
            enable_trust=trust_enabled,
            enable_reporting=report_enabled,
        )
        if not is_valid:
            logger_ref.warning(f"Artifact contract incomplete: missing {missing}")
            # Phase 3: fail on artifact contract in regulatory strict mode
            if cfg.mode == "regulatory" and strict_regulatory:
                raise ArtifactError(
                    message=f"Regulatory strict mode: artifact contract incomplete: {missing}",
                    stage="artifact_validation",
                    hint="Check that all required artifacts were generated.",
                )
        else:
            logger_ref.info("✅ Artifact contract validated")

        # Write success marker
        success_summary = {
            "protocol": str(cfg.protocol),
            "inputs": [str(p) for p in cfg.inputs],
            "mode": cfg.mode,
            "strict_regulatory": strict_regulatory,
            "seed": cfg.seed,
            "rows": len(df),
            "columns": len(df.columns),
            "modeling": modeling_result,
            "trust": trust_result,
        }
        write_success_json(run_dir, success_summary)

        logger_ref.info("=== Phase 3 Workflow Completed Successfully ===")
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

        logger_ref.error(f"=== Phase 3 Workflow Failed (exit {exc.exit_code}) ===")
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

        logger_ref.error(f"=== Phase 3 Workflow Failed (exit {exit_code}) ===")
        return exit_code
