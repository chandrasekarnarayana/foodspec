"""Phase 3: End-to-end real pipeline execution tests.

Tests for:
- Real preprocessing, features, modeling execution
- Real trust stack and reporting
- Strict regulatory enforcement (mandatory QC, trust, reporting)
- Backward compatibility with Phase 1
- Artifact contract Phase 3
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from foodspec.workflow.config import WorkflowConfig
from foodspec.workflow.errors import EXIT_SUCCESS
from foodspec.workflow.phase1_orchestrator import run_workflow
from foodspec.workflow.phase3_orchestrator import run_workflow_phase3


@pytest.fixture
def good_csv(tmp_path):
    """Create good quality test data."""
    df = pd.DataFrame({
        "feature_1": list(range(20)),
        "feature_2": list(range(20, 40)),
        "feature_3": list(range(40, 60)),
        "feature_4": list(range(60, 80)),
        "label": ["a"] * 10 + ["b"] * 10,
    })
    csv_path = tmp_path / "good_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def bad_csv(tmp_path):
    """Create poor quality test data (fails QC)."""
    df = pd.DataFrame({
        "feature_1": [1.0] * 5,  # Too few rows
        "feature_2": [2.0] * 5,
        "label": ["a"] * 4 + ["b"],  # Imbalanced classes
    })
    csv_path = tmp_path / "bad_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# PHASE 3 E2E TESTS (strict behavior)
# ============================================================================


class TestPhase3EndToEnd:
    """End-to-end tests for Phase 3 strict mode."""

    def test_phase3_research_mode_succeeds(self, good_csv, tmp_path):
        """Test that Phase 3 research mode runs successfully with real pipeline."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_research",
            mode="research",
            seed=42,
            enable_modeling=True,
            allow_placeholder_trust=True,  # Task A: Allow placeholder for research mode
        )

        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        assert exit_code == EXIT_SUCCESS

        # Check artifacts (PART C - restored)
        run_dir = cfg.output_dir
        assert (run_dir / "manifest.json").exists(), "Missing manifest.json"
        assert (run_dir / "success.json").exists(), "Missing success.json"
        assert (run_dir / "logs" / "run.log").exists(), "Missing run.log"
        assert (run_dir / "report" / "index.html").exists(), "Missing report"

        # Check that manifest has resolved model
        manifest_data = json.loads((run_dir / "manifest.json").read_text())
        assert "resolved" in manifest_data["artifacts"]
        assert manifest_data["artifacts"]["resolved"]["model"] == "logreg"

    def test_phase3_regulatory_strict_with_qc_pass(self, good_csv, tmp_path):
        """Test that Phase 3 regulatory strict enforces QC and passes with good data."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_regulatory_pass",
            mode="regulatory",
            seed=42,
            enable_modeling=True,  # Regulatory mode requires modeling for trust
            allow_placeholder_trust=True,  # Task A: Allow placeholder for this test
        )

        # Regulatory strict mode should succeed with good data
        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        assert exit_code == EXIT_SUCCESS

        # Verify QC artifacts present (PART C - restored)
        run_dir = cfg.output_dir
        assert (run_dir / "artifacts" / "qc_results.json").exists(), "Missing qc_results.json"

        # Check QC results structure
        qc_data = json.loads((run_dir / "artifacts" / "qc_results.json").read_text())
        assert qc_data["passed"] is True
        assert "gates" in qc_data

    def test_phase3_regulatory_strict_with_qc_fail(self, bad_csv, tmp_path):
        """Test that Phase 3 regulatory strict enforces QC and fails with bad data."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[bad_csv],
            output_dir=tmp_path / "run_regulatory_fail",
            mode="regulatory",
            seed=42,
            enable_modeling=True,  # Enable modeling (though will fail at QC)
        )

        # Regulatory strict mode should fail with bad QC data (exit 7 for QC)
        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        assert exit_code != EXIT_SUCCESS
        assert exit_code == 7, f"Expected exit code 7 (QC failure), got {exit_code}"  # Specific exit code for QC failure

        # Verify error was recorded (PART C - restored)
        run_dir = cfg.output_dir
        assert (run_dir / "error.json").exists(), "Missing error.json"

        error_data = json.loads((run_dir / "error.json").read_text())
        assert "QC" in error_data.get("message", ""), "Error message should mention QC"
        assert error_data.get("exit_code") == 7, "Exit code in error.json should be 7"

    def test_phase3_placeholder_trust_rejected_in_strict_mode(self, good_csv, tmp_path):
        """Task A: Test that placeholder trust is rejected in strict regulatory mode by default.

        Task A: Explicit placeholder governance
        - Strict regulatory mode REJECTS placeholder trust by default (allow_placeholder_trust=False)
        - Should raise TrustError with exit code 6
        - User must explicitly enable with --allow-placeholder-trust flag
        """
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_placeholder_rejected",
            mode="regulatory",
            seed=42,
            enable_modeling=True,
            enable_trust=True,
            allow_placeholder_trust=False,  # Task A: explicit rejection (default)
        )

        # Task A: Strict regulatory should REJECT placeholder trust (exit 6)
        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        assert exit_code != EXIT_SUCCESS
        assert exit_code == 6, f"Task A: Expected exit 6 (TrustError), got {exit_code}"

        # Verify error was recorded with trust-specific message
        run_dir = cfg.output_dir
        assert (run_dir / "error.json").exists(), "Missing error.json"

        error_data = json.loads((run_dir / "error.json").read_text())
        assert "Placeholder" in error_data.get("message", ""), (
            "Task A: Error should mention placeholder trust rejection"
        )
        assert error_data.get("exit_code") == 6, "Exit code should be 6 for TrustError"

    def test_phase3_placeholder_trust_allowed_with_flag(self, good_csv, tmp_path):
        """Task A: Test that placeholder trust is accepted with --allow-placeholder-trust flag.

        Task A: For development/testing, --allow-placeholder-trust allows placeholder
        - Runs successfully (exit 0)
        - Logs warning about placeholder in strict mode
        - Marks trust_stack.json with "implementation": "placeholder"
        """
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_placeholder_allowed",
            mode="regulatory",
            seed=42,
            enable_modeling=True,
            enable_trust=True,
            allow_placeholder_trust=True,  # Task A: explicit allow
        )

        # Task A: Strict regulatory with allow_placeholder_trust=True should succeed
        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        assert exit_code == EXIT_SUCCESS, (
            f"Task A: With --allow-placeholder-trust, should succeed. Got exit {exit_code}"
        )

        # Verify trust_stack.json has "implementation": "placeholder"
        run_dir = cfg.output_dir
        trust_path = run_dir / "artifacts" / "trust_stack.json"
        assert trust_path.exists(), "Missing trust_stack.json"

        trust_data = json.loads(trust_path.read_text())
        assert trust_data.get("implementation") == "placeholder", (
            "Task A: trust_stack.json should mark 'implementation': 'placeholder'"
        )
        assert "capabilities" in trust_data, (
            "Task A: trust_stack.json should include 'capabilities' field"
        )

    def test_phase3_regulatory_strict_mandatory_trust(self, good_csv, tmp_path):
        """Test that Phase 3 regulatory strict forces trust stack enabled."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_regulatory_trust",
            mode="regulatory",
            seed=42,
            enable_modeling=True,  # Modeling required for trust
            enable_trust=False,  # Try to disable trust (will be overridden)
            allow_placeholder_trust=True,  # Task A: Allow placeholder for this test
        )

        # Regulatory strict mode should force trust to be enabled
        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        # Should succeed (trust is forced on)
        assert exit_code == EXIT_SUCCESS

    def test_phase3_regulatory_strict_mandatory_reporting(self, good_csv, tmp_path):
        """Test that Phase 3 regulatory strict forces reporting enabled."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_regulatory_report",
            mode="regulatory",
            seed=42,
            enable_modeling=True,  # Modeling required for complete workflow
            enable_reporting=False,  # Try to disable reporting (will be overridden)
            allow_placeholder_trust=True,  # Task A: Allow placeholder for this test
        )

        # Regulatory strict mode should force reporting to be enabled
        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        # Should succeed (reporting is forced on)
        assert exit_code == EXIT_SUCCESS

        # Verify report was created
        run_dir = cfg.output_dir
        assert (run_dir / "report" / "index.html").exists()

    def test_phase3_research_advisory_qc(self, good_csv, tmp_path):
        """Test that Phase 3 research mode has advisory QC (no enforcement)."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_research_advisory",
            mode="research",
            seed=42,
            enable_modeling=False,  # Skip modeling
            enforce_qc=False,  # Advisory QC
        )

        # Research mode should pass even with advisory QC
        exit_code = run_workflow_phase3(cfg, strict_regulatory=False)

        assert exit_code == EXIT_SUCCESS

    def test_phase3_unapproved_model_rejected(self, good_csv, tmp_path):
        """Test that Phase 3 regulatory rejects unapproved models."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_unapproved_model",
            mode="regulatory",
            seed=42,
            enable_modeling=False,  # Skip modeling to avoid other errors
            model="WeirdCustomModel",  # Not in approved registry
        )

        # Regulatory mode should reject unapproved model (exit 4)
        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        assert exit_code == 4  # Model approval error

    def test_phase3_artifact_contract_strict(self, good_csv, tmp_path):
        """Test that Phase 3 regulatory strict enforces artifact contract."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_artifacts",
            mode="regulatory",
            seed=42,
            enable_modeling=True,  # Modeling required for complete artifact contract
            allow_placeholder_trust=True,  # Task A: Allow placeholder for this test
        )

        exit_code = run_workflow_phase3(cfg, strict_regulatory=True)

        assert exit_code == EXIT_SUCCESS

        run_dir = cfg.output_dir

        # Phase 3 strict regulatory must have all these artifacts (PART C - restored)
        assert (run_dir / "manifest.json").exists(), "Missing manifest.json"
        assert (run_dir / "success.json").exists(), "Missing success.json"
        assert (run_dir / "logs" / "run.log").exists(), "Missing run.log"
        assert (run_dir / "artifacts" / "qc_results.json").exists(), "Missing QC results"
        assert (run_dir / "report" / "index.html").exists(), "Missing report"
        assert (run_dir / "tables" / "preprocessed.csv").exists(), "Missing preprocessed data"
        assert (run_dir / "tables" / "features.csv").exists(), "Missing features"


# ============================================================================
# BACKWARD COMPATIBILITY TESTS (Phase 1 still works)
# ============================================================================


class TestPhase1BackwardCompat:
    """Ensure Phase 1 tests still work unchanged."""

    def test_phase1_workflow_still_works(self, good_csv, tmp_path):
        """Test that Phase 1 workflow-run command still works."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_phase1_compat",
            mode="research",
            seed=42,
            enable_modeling=False,  # Phase 1 stub behavior
        )

        # Phase 1 command should still work
        exit_code = run_workflow(cfg)

        assert exit_code == EXIT_SUCCESS

    def test_phase1_regulatory_relaxed_no_strict(self, good_csv, tmp_path):
        """Test that Phase 1 regulatory mode (no strict) remains relaxed."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_phase1_regulatory_relaxed",
            mode="regulatory",
            seed=42,
            enable_modeling=False,  # Phase 1 stub behavior
        )

        # Phase 1 regulatory should succeed without enforcing trust/reporting
        exit_code = run_workflow(cfg)

        assert exit_code == EXIT_SUCCESS


# ============================================================================
# PHASE 3 VS PHASE 1 CONTRAST TESTS
# ============================================================================


class TestPhase3VsPhase1Contrast:
    """Tests showing the difference between Phase 1 and Phase 3."""

    def test_phase1_vs_phase3_regulatory_semantics(self, bad_csv, tmp_path):
        """Test that Phase 1 regulatory is relaxed, Phase 3 is strict."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        # Phase 1 regulatory with bad data: should pass (no QC enforcement)
        cfg_p1 = WorkflowConfig(
            protocol=protocol_path,
            inputs=[bad_csv],
            output_dir=tmp_path / "p1_regulatory_bad",
            mode="regulatory",
            enable_modeling=False,
        )
        exit_p1 = run_workflow(cfg_p1)

        # Phase 3 strict regulatory with bad data: should fail (QC enforced)
        cfg_p3 = WorkflowConfig(
            protocol=protocol_path,
            inputs=[bad_csv],
            output_dir=tmp_path / "p3_regulatory_bad",
            mode="regulatory",
            enable_modeling=False,
        )
        exit_p3 = run_workflow_phase3(cfg_p3, strict_regulatory=True)

        # Phase 1 passes (relaxed), Phase 3 fails (strict)
        assert exit_p1 == EXIT_SUCCESS, "Phase 1 should pass with relaxed regulatory"
        assert exit_p3 != EXIT_SUCCESS, "Phase 3 should fail with strict regulatory"

    def test_phase3_compat_mode_relaxed(self, bad_csv, tmp_path):
        """Test that Phase 3 with strict_regulatory=False matches Phase 1."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[bad_csv],
            output_dir=tmp_path / "p3_compat_relaxed",
            mode="regulatory",
            enable_modeling=False,
        )

        # Phase 3 with compat mode should behave like Phase 1
        exit_code = run_workflow_phase3(cfg, strict_regulatory=False)

        # Should pass (compat mode is relaxed)
        assert exit_code == EXIT_SUCCESS, "Phase 3 compat should match Phase 1"


# ============================================================================
# MODELING INTEGRATION TESTS
# ============================================================================


class TestPhase3ModelingIntegration:
    """Tests for real modeling integration."""

    def test_phase3_modeling_metrics(self, good_csv, tmp_path):
        """Test that Phase 3 produces real modeling metrics."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[good_csv],
            output_dir=tmp_path / "run_metrics",
            mode="research",
            seed=42,
            enable_modeling=True,
            model="logreg",
            scheme="kfold",
        )

        exit_code = run_workflow_phase3(cfg, strict_regulatory=False)

        assert exit_code == EXIT_SUCCESS

        # Check manifest contains modeling results (PART C - restored)
        manifest_path = cfg.output_dir / "manifest.json"
        assert manifest_path.exists(), "Missing manifest.json"

        manifest_data = json.loads(manifest_path.read_text())
        assert "artifacts" in manifest_data
        modeling = manifest_data["artifacts"].get("modeling", {})
        assert modeling.get("status") == "success", "Modeling should succeed"
        assert "metrics" in modeling, "Modeling should produce metrics"

        # Check metrics file exists
        metrics_path = cfg.output_dir / "artifacts" / "metrics.json"
        assert metrics_path.exists(), "Missing metrics.json"

        metrics_data = json.loads(metrics_path.read_text())
        assert "accuracy" in metrics_data, "Metrics should include accuracy"

    def test_phase3_modeling_with_groups(self, good_csv, tmp_path):
        """Test that Phase 3 modeling respects group columns."""
        protocol_path = Path("tests/fixtures/minimal_protocol_phase3.yaml")
        if not protocol_path.exists():
            pytest.skip("Phase 3 protocol fixture not found")

        # Create data with groups (need enough groups for CV)
        df = pd.read_csv(good_csv)
        # Create 5 groups with 4 samples each to support 5-fold group CV
        df["group"] = ["g1"] * 4 + ["g2"] * 4 + ["g3"] * 4 + ["g4"] * 4 + ["g5"] * 4
        grouped_csv = tmp_path / "grouped_data.csv"
        df.to_csv(grouped_csv, index=False)

        cfg = WorkflowConfig(
            protocol=protocol_path,
            inputs=[grouped_csv],
            output_dir=tmp_path / "run_grouped",
            mode="research",
            seed=42,
            enable_modeling=True,
            model="logreg",
            group_col="group",
        )

        exit_code = run_workflow_phase3(cfg, strict_regulatory=False)

        assert exit_code == EXIT_SUCCESS


# ============================================================================
# TASK D: CONTRACT DIGEST LOCK TESTS
# ============================================================================


class TestTaskDContractDigestLock:
    """Tests for Task D: prevent contract drift by locking digest hash."""

    def test_contract_v3_digest_lock(self):
        """Test that contract_v3.json digest is locked and detectable.

        Task D: Lock digest against drift.
        - Compute current digest
        - Assert it matches expected value
        - If changed, test fails and requires intentional update
        """
        from foodspec.workflow.artifact_contract import ArtifactContract

        # Load contract v3
        contract_dict = ArtifactContract._load_contract(version="v3")
        current_digest = ArtifactContract.compute_digest(contract_dict)

        # Task D: Expected digest (locked after Task B completed - success.json removed)
        EXPECTED_DIGEST = "61f345763075100e57f0ea0cbb9e098aabae15549aad43933a230ce1c4a9154f"

        # If this assertion fails:
        # 1. Check if contract_v3.json was intentionally changed
        # 2. If yes, update EXPECTED_DIGEST with new value from test output
        # 3. If no, revert contract_v3.json changes to restore expected state
        assert current_digest == EXPECTED_DIGEST, (
            f"Contract digest mismatch (Task D drift detection). "
            f"Expected: {EXPECTED_DIGEST}, got: {current_digest}. "
            f"If contract_v3.json was intentionally modified, update EXPECTED_DIGEST. "
            f"Otherwise, revert contract_v3.json to prevent unintentional drift."
        )

    def test_contract_v3_has_implementation_fields(self):
        """Test that contract validates new Task A implementation fields.

        Task A/D: Trust stack must return implementation + capabilities.
        """
        from foodspec.workflow.artifact_contract import ArtifactContract

        # Load contract v3
        contract_dict = ArtifactContract._load_contract(version="v3")

        # Contract should validate artifacts (not prescribe trust fields)
        # But trust_stack.json is in required_trust
        assert "required_trust" in contract_dict
        assert "artifacts/trust_stack.json" in contract_dict["required_trust"]


# ============================================================================
# CLI INTEGRATION TESTS (if needed)
# ============================================================================


class TestPhase3CLIIntegration:
    """Tests for Phase 3 CLI command integration."""

    def test_phase3_cli_command_exists(self):
        """Test that Phase 3 CLI command is registered."""
        from foodspec.cli.commands.workflow import workflow_app

        # Check that run-strict command exists
        commands = [cmd.name for cmd in workflow_app.registered_commands]
        assert "run-strict" in commands, "Phase 3 'run-strict' command not registered"

    def test_phase3_cli_allow_placeholder_trust_flag_exists(self):
        """Test that --allow-placeholder-trust flag is available (Task A).

        Task A: CLI should support --allow-placeholder-trust for development.
        """
        import inspect

        from foodspec.cli.commands.workflow import run_phase3_workflow

        # Check that run_phase3_workflow has allow_placeholder_trust parameter
        sig = inspect.signature(run_phase3_workflow)
        params = sig.parameters
        assert "allow_placeholder_trust" in params, (
            "Task A: --allow-placeholder-trust flag not found in CLI"
        )

    def test_phase3_cli_phase_selection_flag_exists(self):
        """Test that --phase flag is available (Task C).

        Task C: CLI should support --phase {1,2,3} for phase selection.
        """
        import inspect

        from foodspec.cli.commands.workflow import run_phase3_workflow

        # Check that run_phase3_workflow has phase parameter
        sig = inspect.signature(run_phase3_workflow)
        params = sig.parameters
        assert "phase" in params, (
            "Task C: --phase flag not found in CLI"
        )
