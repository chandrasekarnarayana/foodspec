"""Phase 1 Orchestrator Tests.

Tests for the minimal guaranteed E2E workflow with:
- Config validation
- Fingerprinting
- Error handling + exit codes
- Artifact contract
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest

from foodspec.workflow.config import WorkflowConfig
from foodspec.workflow.fingerprint import (
    Manifest,
    compute_dataset_fingerprint,
    compute_protocol_fingerprint,
)
from foodspec.workflow.errors import (
    WorkflowError,
    ValidationError,
    ProtocolError,
    ModelingError,
    write_error_json,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
    EXIT_PROTOCOL_ERROR,
    EXIT_ARTIFACT_ERROR,
)
from foodspec.workflow.artifact_contract import (
    ArtifactContract,
    write_success_json,
)
from foodspec.workflow.phase1_orchestrator import run_workflow


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_csv():
    """Path to minimal test CSV."""
    return Path(__file__).parent / "fixtures" / "minimal.csv"


@pytest.fixture
def minimal_protocol():
    """Path to minimal test protocol."""
    return Path(__file__).parent / "fixtures" / "minimal_protocol.yaml"


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Temporary run directory."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    return run_dir


# ============================================================================
# Unit Tests: WorkflowConfig
# ============================================================================

def test_workflow_config_validation_success(minimal_csv, minimal_protocol):
    """WorkflowConfig.validate() returns True for valid config."""
    cfg = WorkflowConfig(
        protocol=minimal_protocol,
        inputs=[minimal_csv],
        mode="research",
    )
    is_valid, errors = cfg.validate()
    assert is_valid is True
    assert errors == []


def test_workflow_config_validation_missing_protocol(minimal_csv, tmp_path):
    """WorkflowConfig.validate() fails for missing protocol."""
    cfg = WorkflowConfig(
        protocol=tmp_path / "nonexistent.yaml",
        inputs=[minimal_csv],
    )
    is_valid, errors = cfg.validate()
    assert is_valid is False
    assert any("not found" in e.lower() for e in errors)


def test_workflow_config_validation_missing_input(minimal_protocol, tmp_path):
    """WorkflowConfig.validate() fails for missing input."""
    cfg = WorkflowConfig(
        protocol=minimal_protocol,
        inputs=[tmp_path / "nonexistent.csv"],
    )
    is_valid, errors = cfg.validate()
    assert is_valid is False
    assert any("not found" in e.lower() for e in errors)


def test_workflow_config_validation_invalid_mode(minimal_csv, minimal_protocol):
    """WorkflowConfig.validate() fails for invalid mode."""
    cfg = WorkflowConfig(
        protocol=minimal_protocol,
        inputs=[minimal_csv],
        mode="invalid",
    )
    is_valid, errors = cfg.validate()
    assert is_valid is False
    assert any("mode" in e.lower() for e in errors)


def test_workflow_config_summary(minimal_csv, minimal_protocol):
    """WorkflowConfig.summary() returns readable string."""
    cfg = WorkflowConfig(
        protocol=minimal_protocol,
        inputs=[minimal_csv],
        mode="research",
        seed=42,
    )
    summary = cfg.summary()
    assert "Protocol" in summary
    assert "research" in summary
    assert "42" in summary


# ============================================================================
# Unit Tests: Fingerprinting
# ============================================================================

def test_compute_dataset_fingerprint(minimal_csv):
    """compute_dataset_fingerprint() returns expected keys."""
    fp = compute_dataset_fingerprint(minimal_csv)
    
    assert "sha256" in fp
    assert "rows" in fp
    assert "columns" in fp
    assert "missing_per_column" in fp
    assert fp["rows"] == 10
    assert "label" in fp["columns"]
    assert fp["missing_per_column"]["label"] == 0.0


def test_compute_protocol_fingerprint(minimal_protocol):
    """compute_protocol_fingerprint() returns expected keys."""
    fp = compute_protocol_fingerprint(minimal_protocol)
    
    assert "sha256" in fp
    assert "path" in fp
    assert "size_bytes" in fp
    assert fp["sha256"] is not None
    assert len(fp["sha256"]) == 64  # SHA256 hex


def test_manifest_build(minimal_csv, minimal_protocol):
    """Manifest.build() creates valid manifest."""
    manifest = Manifest.build(
        protocol_path=minimal_protocol,
        input_paths=[minimal_csv],
        seed=42,
        mode="research",
    )
    
    assert manifest.foodspec_version is not None
    assert manifest.python_version is not None
    assert manifest.seed == 42
    assert manifest.mode == "research"
    assert len(manifest.dataset_fingerprints) == 1
    assert manifest.protocol_fingerprint["sha256"] is not None


def test_manifest_finalize():
    """Manifest.finalize() sets end_time and duration."""
    manifest = Manifest.build(
        protocol_path=Path("dummy.yaml"),
        input_paths=[],
        seed=42,
    )
    manifest.finalize()
    
    assert manifest.end_time != ""
    assert manifest.duration_seconds is not None
    assert manifest.duration_seconds >= 0


def test_manifest_save_load(tmp_path, minimal_csv, minimal_protocol):
    """Manifest.save() and .load() work correctly."""
    manifest_path = tmp_path / "manifest.json"
    
    manifest = Manifest.build(
        protocol_path=minimal_protocol,
        input_paths=[minimal_csv],
        seed=42,
    )
    manifest.finalize()
    manifest.save(manifest_path)
    
    assert manifest_path.exists()
    
    # Load and verify
    loaded = Manifest.load(manifest_path)
    assert loaded.seed == 42
    assert loaded.mode == "research"


# ============================================================================
# Unit Tests: Error Handling
# ============================================================================

def test_validation_error_exit_code():
    """ValidationError has correct exit code."""
    err = ValidationError(
        message="Test error",
        stage="data_loading",
    )
    assert err.exit_code == EXIT_VALIDATION_ERROR


def test_protocol_error_exit_code():
    """ProtocolError has correct exit code."""
    err = ProtocolError(
        message="Test error",
        stage="protocol_loading",
    )
    assert err.exit_code == EXIT_PROTOCOL_ERROR


def test_error_to_dict():
    """WorkflowError.to_dict() returns JSON-serializable dict."""
    err = ValidationError(
        message="Test error",
        stage="validation",
        hint="Fix the input.",
    )
    data = err.to_dict()
    
    assert data["error_type"] == "ValidationError"
    assert data["message"] == "Test error"
    assert data["stage"] == "validation"
    assert data["hint"] == "Fix the input."
    assert data["exit_code"] == EXIT_VALIDATION_ERROR


def test_write_error_json(tmp_path):
    """write_error_json() creates error.json with correct structure."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    
    exc = ValidationError(
        message="Test error",
        stage="data_loading",
        hint="Check input files.",
    )
    
    error_path = write_error_json(
        run_dir,
        exc,
        stage="data_loading",
        exit_code=EXIT_VALIDATION_ERROR,
        hint="Check input files.",
    )
    
    assert error_path.exists()
    
    with error_path.open() as f:
        error_data = json.load(f)
    
    assert error_data["error_type"] == "ValidationError"
    assert error_data["message"] == "Test error"
    assert error_data["exit_code"] == EXIT_VALIDATION_ERROR


# ============================================================================
# Unit Tests: Artifact Contract
# ============================================================================

def test_artifact_contract_success_validation_missing_files(tmp_path):
    """ArtifactContract.validate_success() fails when required files missing."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    
    is_valid, missing = ArtifactContract.validate_success(run_dir)
    assert is_valid is False
    assert "manifest.json" in missing
    assert "logs/run.log" in missing


def test_artifact_contract_success_validation_all_present(tmp_path):
    """ArtifactContract.validate_success() passes when all files present."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "manifest.json").touch()
    (run_dir / "logs" / "run.log").touch()
    (run_dir / "success.json").touch()
    
    is_valid, missing = ArtifactContract.validate_success(run_dir)
    assert is_valid is True
    assert missing == []


def test_artifact_contract_failure_validation(tmp_path):
    """ArtifactContract.validate_failure() requires error.json."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "manifest.json").touch()
    (run_dir / "logs" / "run.log").touch()
    
    is_valid, missing = ArtifactContract.validate_failure(run_dir)
    assert is_valid is False
    assert "error.json" in missing


def test_write_success_json(tmp_path):
    """write_success_json() creates success.json with correct structure."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    
    summary = {
        "protocol": "test.yaml",
        "inputs": ["data.csv"],
        "rows": 100,
    }
    
    success_path = write_success_json(run_dir, summary)
    
    assert success_path.exists()
    
    with success_path.open() as f:
        data = json.load(f)
    
    assert data["status"] == "success"
    assert data["summary"]["rows"] == 100


# ============================================================================
# Integration Tests
# ============================================================================

def test_workflow_run_minimal_research_mode(minimal_csv, minimal_protocol, tmp_path):
    """Full workflow execution in research mode."""
    cfg = WorkflowConfig(
        protocol=minimal_protocol,
        inputs=[minimal_csv],
        output_dir=tmp_path / "run1",
        mode="research",
        seed=42,
        enable_modeling=False,  # Skip modeling in Phase 1
        verbose=True,
    )
    
    exit_code = run_workflow(cfg)
    
    assert exit_code == EXIT_SUCCESS
    
    # Verify artifacts
    run_dir = tmp_path / "run1"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "logs" / "run.log").exists()
    assert (run_dir / "success.json").exists()
    
    # Verify manifest content
    with (run_dir / "manifest.json").open() as f:
        manifest = json.load(f)
    assert manifest["seed"] == 42
    assert manifest["mode"] == "research"
    assert len(manifest["dataset_fingerprints"]) == 1


def test_workflow_run_missing_input(minimal_protocol, tmp_path):
    """Workflow fails gracefully when input missing."""
    cfg = WorkflowConfig(
        protocol=minimal_protocol,
        inputs=[tmp_path / "nonexistent.csv"],
        output_dir=tmp_path / "run_error",
        mode="research",
    )
    
    exit_code = run_workflow(cfg)
    
    assert exit_code != EXIT_SUCCESS
    
    # Verify error artifact
    run_dir = tmp_path / "run_error"
    assert (run_dir / "error.json").exists()
    
    with (run_dir / "error.json").open() as f:
        error_data = json.load(f)
    assert error_data["exit_code"] in (EXIT_VALIDATION_ERROR, EXIT_PROTOCOL_ERROR)


def test_workflow_run_missing_protocol(minimal_csv, tmp_path):
    """Workflow fails gracefully when protocol missing."""
    cfg = WorkflowConfig(
        protocol=tmp_path / "nonexistent_protocol.yaml",
        inputs=[minimal_csv],
        output_dir=tmp_path / "run_error2",
        mode="research",
    )
    
    exit_code = run_workflow(cfg)
    
    assert exit_code != EXIT_SUCCESS
    
    # Verify error artifact
    run_dir = tmp_path / "run_error2"
    assert (run_dir / "error.json").exists()


def test_workflow_run_creates_logs(minimal_csv, minimal_protocol, tmp_path):
    """Workflow creates proper log files."""
    cfg = WorkflowConfig(
        protocol=minimal_protocol,
        inputs=[minimal_csv],
        output_dir=tmp_path / "run_logs",
        mode="research",
        enable_modeling=False,
        verbose=True,
    )
    
    run_workflow(cfg)
    
    run_dir = tmp_path / "run_logs"
    log_file = run_dir / "logs" / "run.log"
    jsonl_file = run_dir / "logs" / "run.jsonl"
    
    assert log_file.exists()
    assert jsonl_file.exists()
    
    # Check log content
    log_content = log_file.read_text()
    assert "Workflow started" in log_content or "started" in log_content.lower()
    
    # Check JSONL structure
    jsonl_content = jsonl_file.read_text()
    lines = jsonl_content.strip().split("\n")
    assert len(lines) > 0
    
    for line in lines:
        data = json.loads(line)
        assert "timestamp" in data
        assert "level" in data


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("mode", ["research", "regulatory"])
def test_workflow_modes(mode, minimal_csv, minimal_protocol, tmp_path):
    """Workflow runs with different modes."""
    cfg = WorkflowConfig(
        protocol=minimal_protocol,
        inputs=[minimal_csv],
        output_dir=tmp_path / f"run_{mode}",
        mode=mode,
        seed=42,
        enable_modeling=False,
    )
    
    exit_code = run_workflow(cfg)
    
    # In Phase 1, both modes should succeed
    assert exit_code == EXIT_SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
