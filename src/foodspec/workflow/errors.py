"""Workflow error handling with exit code contracts.

Defines exception classes for each phase of the workflow with corresponding
exit codes, and provides error.json serialization helpers.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Exit code contract
EXIT_SUCCESS = 0
EXIT_CLI_ERROR = 2
EXIT_VALIDATION_ERROR = 3
EXIT_PROTOCOL_ERROR = 4
EXIT_MODELING_ERROR = 5
EXIT_TRUST_ERROR = 6
EXIT_QC_ERROR = 7
EXIT_REPORTING_ERROR = 8
EXIT_ARTIFACT_ERROR = 9


@dataclass
class WorkflowError(Exception):
    """Base exception for workflow errors.
    
    All workflow errors are caught and serialized to error.json with
    exit code, message, hints, and traceback.
    """

    message: str
    """Human-readable error message."""

    exit_code: int = EXIT_SUCCESS
    """Exit code to return to shell."""

    stage: str = "unknown"
    """Which workflow stage failed (e.g. 'validation', 'modeling')."""

    hint: str = ""
    """Suggestion for how to fix the error."""

    traceback_str: str = ""
    """Captured Python traceback."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional context (stage-specific)."""

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to JSON-serializable dict."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "stage": self.stage,
            "exit_code": self.exit_code,
            "hint": self.hint,
            "traceback": self.traceback_str,
            "details": self.details,
        }


class CLIError(WorkflowError):
    """Error in CLI argument parsing or validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = EXIT_CLI_ERROR


class ValidationError(WorkflowError):
    """Error in input data validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = EXIT_VALIDATION_ERROR


class ProtocolError(WorkflowError):
    """Error in protocol loading or validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = EXIT_PROTOCOL_ERROR


class ModelingError(WorkflowError):
    """Error in model training or prediction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = EXIT_MODELING_ERROR


class TrustError(WorkflowError):
    """Error in trust stack (calibration, conformal, etc.)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = EXIT_TRUST_ERROR


class QCError(WorkflowError):
    """Error in QC gate enforcement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = EXIT_QC_ERROR


class ReportingError(WorkflowError):
    """Error in report generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = EXIT_REPORTING_ERROR


class ArtifactError(WorkflowError):
    """Error in artifact contract validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = EXIT_ARTIFACT_ERROR


def write_error_json(
    run_dir: Path,
    error: Exception,
    stage: str,
    exit_code: Optional[int] = None,
    hint: str = "",
) -> Path:
    """Write error.json artifact to run directory.
    
    Parameters
    ----------
    run_dir : Path
        Run directory (containing logs/).
    error : Exception
        Exception that was raised.
    stage : str
        Workflow stage where error occurred.
    exit_code : Optional[int]
        Exit code (inferred from exception if WorkflowError).
    hint : str
        Human-friendly suggestion for fixing.
    
    Returns
    -------
    Path
        Path to written error.json file.
    """
    # Create error dict
    error_dict = {
        "error_type": error.__class__.__name__,
        "message": str(error),
        "stage": stage,
        "exit_code": exit_code or 1,
        "hint": hint,
        "traceback": traceback.format_exc(),
    }

    # If it's a WorkflowError, use its structure
    if isinstance(error, WorkflowError):
        error_dict.update(error.to_dict())

    # Write to run_dir/error.json
    error_path = run_dir / "error.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    with error_path.open("w") as f:
        json.dump(error_dict, f, indent=2, default=str)

    return error_path


def classify_error_type(exc: Exception) -> tuple[WorkflowError, int]:
    """Classify an exception into WorkflowError + exit code.
    
    If already a WorkflowError, returns as-is.
    Otherwise, tries to infer based on exception type.
    
    Parameters
    ----------
    exc : Exception
        Exception to classify.
    
    Returns
    -------
    (WorkflowError, exit_code)
        Classified error and exit code.
    """
    if isinstance(exc, WorkflowError):
        return exc, exc.exit_code

    if isinstance(exc, FileNotFoundError):
        err = ValidationError(
            message=f"File not found: {exc}",
            stage="data_loading",
            hint="Check file paths and ensure input files exist.",
        )
        return err, EXIT_VALIDATION_ERROR

    if isinstance(exc, ValueError):
        err = ValidationError(
            message=f"Invalid value: {exc}",
            stage="validation",
            hint="Check input data types and ranges.",
        )
        return err, EXIT_VALIDATION_ERROR

    if isinstance(exc, ImportError):
        err = ProtocolError(
            message=f"Import failed: {exc}",
            stage="protocol_loading",
            hint="Ensure all required packages are installed.",
        )
        return err, EXIT_PROTOCOL_ERROR

    # Default to generic error
    err = WorkflowError(
        message=f"Unexpected error: {exc}",
        stage="unknown",
        exit_code=1,
        hint="Check logs for details.",
    )
    return err, 1
