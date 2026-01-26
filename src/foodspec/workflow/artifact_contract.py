"""Artifact contract validation for workflow runs.

Enforces that required artifacts exist and validates their structure
for both success and failure paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json


class ArtifactContract:
    """Defines which artifacts must exist at different stages."""

    # Mandatory for all runs
    REQUIRED_ALWAYS = {
        "manifest.json": "Execution manifest with versions, fingerprints, timing",
        "logs/run.log": "Human-readable log file",
    }

    # On success
    REQUIRED_SUCCESS = {
        "success.json": "Success marker with status and summary",
    }
    
    # Phase 2: QC artifacts (conditional)
    REQUIRED_QC = {
        "artifacts/qc_results.json": "QC gate results (data integrity, spectral quality, model reliability)",
    }
    
    # Phase 2: Trust artifacts (conditional - when enable_trust=True)
    REQUIRED_TRUST = {
        "artifacts/trust_stack.json": "Trust stack results (calibration, conformal, abstention)",
    }
    
    # Phase 2: Reporting artifacts (conditional - when enable_reporting=True)
    REQUIRED_REPORTING = {
        "artifacts/report.html": "HTML report",
    }

    # On failure
    REQUIRED_FAILURE = {
        "error.json": "Error details, hints, and exit code",
    }

    # Optional phase 1
    OPTIONAL_PHASE1 = {
        "logs/run.jsonl": "Structured JSON logging (one entry per line)",
        "protocol_snapshot.yaml": "Copy of protocol YAML",
        "logs/debug.log": "DEBUG-level logs",
    }

    @classmethod
    def validate_success(
        cls,
        run_dir: Path,
        enforce_qc: bool = False,
        enable_trust: bool = False,
        enable_reporting: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Validate required artifacts for successful run.

        Parameters
        ----------
        run_dir : Path
            Run directory to validate.
        enforce_qc : bool
            If True, QC artifacts are required.
        enable_trust : bool
            If True, trust stack artifacts are required.
        enable_reporting : bool
            If True, reporting artifacts are required.

        Returns
        -------
        (is_valid, missing_files)
            Tuple of boolean and list of missing file paths.
        """
        missing = []
        all_required = {**cls.REQUIRED_ALWAYS, **cls.REQUIRED_SUCCESS}
        
        # Add conditional requirements
        if enforce_qc:
            all_required.update(cls.REQUIRED_QC)
        if enable_trust:
            all_required.update(cls.REQUIRED_TRUST)
        if enable_reporting:
            all_required.update(cls.REQUIRED_REPORTING)

        for file_path in all_required.keys():
            full_path = run_dir / file_path
            if not full_path.exists():
                missing.append(file_path)

        return len(missing) == 0, missing

    @classmethod
    def validate_failure(cls, run_dir: Path) -> Tuple[bool, List[str]]:
        """Validate required artifacts for failed run.

        Parameters
        ----------
        run_dir : Path
            Run directory to validate.

        Returns
        -------
        (is_valid, missing_files)
            Tuple of boolean and list of missing file paths.
        """
        missing = []
        all_required = {**cls.REQUIRED_ALWAYS, **cls.REQUIRED_FAILURE}

        for file_path in all_required.keys():
            full_path = run_dir / file_path
            if not full_path.exists():
                missing.append(file_path)

        return len(missing) == 0, missing

    @classmethod
    def validate_artifacts_exist(
        cls,
        run_dir: Path,
        artifacts: Dict[str, str],
    ) -> Tuple[bool, List[str]]:
        """Validate that recorded artifacts exist on disk.

        Parameters
        ----------
        run_dir : Path
            Run directory.
        artifacts : Dict[str, str]
            Dictionary mapping artifact name to relative path.

        Returns
        -------
        (is_valid, missing)
            Tuple of boolean and list of missing artifact paths.
        """
        missing = []
        for artifact_path in artifacts.values():
            if artifact_path and not (run_dir / artifact_path).exists():
                missing.append(artifact_path)

        return len(missing) == 0, missing

    @classmethod
    def summary(cls, run_dir: Path, is_success: bool) -> str:
        """Return human-readable summary of artifact validation.

        Parameters
        ----------
        run_dir : Path
            Run directory.
        is_success : bool
            Whether run succeeded.

        Returns
        -------
        str
            Summary string.
        """
        is_valid, missing = (
            cls.validate_success(run_dir)
            if is_success
            else cls.validate_failure(run_dir)
        )

        if is_valid:
            return f"✅ Artifact contract validated (all required files present)"
        else:
            return f"❌ Artifact contract FAILED: missing {missing}"


def write_success_json(run_dir: Path, summary: Dict) -> Path:
    """Write success.json to mark successful run.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    summary : Dict
        Summary data to include.

    Returns
    -------
    Path
        Path to written file.
    """
    success_path = run_dir / "success.json"

    success_data = {
        "status": "success",
        "run_dir": str(run_dir),
        "summary": summary,
    }

    with success_path.open("w") as f:
        json.dump(success_data, f, indent=2, default=str)

    return success_path
