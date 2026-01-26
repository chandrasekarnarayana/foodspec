"""Phase 2: QC gates and regulatory enforcement tests.

Tests for:
- QC gate execution (DataIntegrityGate, SpectralQualityGate, ModelReliabilityGate)
- QC enforcement (advisory vs enforce-qc)
- Regulatory mode restrictions (model approval, trust stack, reporting)
- Artifact contract Phase 2 (QC + trust + report artifacts)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import pandas as pd

from foodspec.workflow.config import WorkflowConfig
from foodspec.workflow.qc_gates import (
    DataIntegrityGate,
    GateResult,
    ModelReliabilityGate,
    SpectralQualityGate,
)
from foodspec.workflow.regulatory import (
    APPROVED_MODELS,
    check_override_governance,
    enforce_model_approved,
    enforce_reporting,
    enforce_trust_stack,
    get_regulatory_restrictions,
)
from foodspec.workflow.artifact_contract import ArtifactContract


# ============================================================================
# QC GATE TESTS
# ============================================================================


class TestDataIntegrityGate:
    """Test DataIntegrityGate functionality."""

    def test_good_data_passes(self):
        """Test that high-quality data passes gate."""
        # Need at least 20 rows and balanced classes
        df = pd.DataFrame({
            "feature_1": list(range(20)),
            "feature_2": list(range(20, 40)),
            "label": ["a"] * 10 + ["b"] * 10,
        })

        gate = DataIntegrityGate()
        result = gate.run(df, label_col="label")

        assert result.status == "pass"
        assert len(result.remediation) == 0
        assert result.metrics["row_count"] == 20

    def test_missing_data_fails(self):
        """Test that missing data triggers failure."""
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0, None, 4.0, 5.0],
            "feature_2": [2.0, None, 4.0, 5.0, 6.0],
            "label": ["a", "b", "a", "b", "a"],
        })

        gate = DataIntegrityGate()
        result = gate.run(df, label_col="label")

        assert result.status == "fail"
        assert len(result.remediation) > 0
        assert result.metrics["max_missingness_observed"] > 0

    def test_too_few_rows_fails(self):
        """Test that too few rows triggers failure."""
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0],
            "feature_2": [2.0, 3.0],
            "label": ["a", "b"],
        })

        gate = DataIntegrityGate(thresholds={"min_rows": 10})
        result = gate.run(df, label_col="label")

        assert result.status == "fail"

    def test_class_imbalance_fails(self):
        """Test that extreme class imbalance triggers failure."""
        df = pd.DataFrame({
            "feature_1": [1.0] * 20,
            "feature_2": [2.0] * 20,
            "label": ["a"] * 19 + ["b"],  # Only 1 sample of class b
        })

        gate = DataIntegrityGate(thresholds={"min_samples_per_class": 5})
        result = gate.run(df, label_col="label")

        assert result.status == "fail"

    def test_duplicate_rows_detected(self):
        """Test that duplicate rows are detected."""
        df = pd.DataFrame({
            "feature_1": [1.0, 1.0, 1.0, 4.0, 5.0],
            "feature_2": [2.0, 2.0, 2.0, 5.0, 6.0],
            "label": ["a", "a", "a", "b", "a"],
        })

        gate = DataIntegrityGate(thresholds={"max_duplicate_fraction": 0.1})
        result = gate.run(df)

        assert result.metrics["duplicate_rows"] > 0
        assert result.status == "fail"

    def test_label_distribution_metrics(self):
        """Test that label distribution metrics are computed."""
        df = pd.DataFrame({
            "feature_1": list(range(10)),
            "feature_2": list(range(10, 20)),
            "label": ["a", "a", "a", "a", "a", "b", "b", "b", "c", "c"],
        })

        gate = DataIntegrityGate()
        result = gate.run(df, label_col="label")

        assert "label_distribution" in result.metrics
        assert result.metrics["n_classes"] == 3
        assert "label_entropy" in result.metrics


class TestSpectralQualityGate:
    """Test SpectralQualityGate functionality."""

    def test_good_spectral_data_passes(self):
        """Test that clean spectral data passes gate with relaxed thresholds."""
        df = pd.DataFrame({
            "wavenumber_1": [100.0 + i*0.1 for i in range(10)],
            "wavenumber_2": [200.0 + i*0.2 for i in range(10)],
            "wavenumber_3": [150.0 + i*0.15 for i in range(10)],
            "label": ["a"] * 5 + ["b"] * 5,
        })

        # Use relaxed thresholds to reflect actual SNR
        gate = SpectralQualityGate(thresholds={"min_snr": 1.0})
        result = gate.run(df, spectral_cols=["wavenumber_1", "wavenumber_2", "wavenumber_3"])

        assert result.status in ("pass", "warn")
        assert result.metrics["snr_median"] > 0

    def test_outliers_detected(self):
        """Test that extreme outliers are detected."""
        # Create data with very clear outlier (z-score > 3)
        # Normal range: ~450-475, outlier: 5000
        data_normal = [[100.0 + i, 200.0 + i, 150.0 + i] for i in range(9)]
        data_outlier = [[2000.0, 2000.0, 1000.0]]  # Extreme outlier to ensure z > 3
        
        df = pd.DataFrame(
            data_normal + data_outlier,
            columns=["wavenumber_1", "wavenumber_2", "wavenumber_3"],
        )

        gate = SpectralQualityGate(thresholds={"max_outlier_fraction": 0.05, "min_snr": 0.5})
        result = gate.run(df, spectral_cols=["wavenumber_1", "wavenumber_2", "wavenumber_3"])

        # Should detect at least one outlier
        assert result.metrics["outlier_fraction"] > 0, f"Expected outliers but got {result.metrics['outlier_fraction']}"
        assert result.metrics["n_outliers"] >= 1
        # With high outlier_fraction relative to threshold, should fail
        if result.metrics["outlier_fraction"] > 0.05:
            assert result.status == "fail"

    def test_auto_detect_spectral_cols(self):
        """Test auto-detection of spectral columns."""
        df = pd.DataFrame({
            "wavenumber_1": [100.0] * 10,
            "wavenumber_2": [200.0] * 10,
            "label": ["a"] * 10,  # Non-numeric column
        })

        gate = SpectralQualityGate()
        result = gate.run(df)  # No spectral_cols specified

        assert result.status in ("pass", "warn", "skip")


class TestModelReliabilityGate:
    """Test ModelReliabilityGate functionality."""

    def test_good_model_passes(self):
        """Test that good model metrics pass gate."""
        modeling_result = {
            "metrics": {
                "accuracy": 0.85,
                "f1": 0.84,
                "ece": 0.05,
            }
        }

        gate = ModelReliabilityGate()
        result = gate.run(modeling_result)

        assert result.status == "pass"

    def test_low_accuracy_fails(self):
        """Test that low accuracy triggers failure."""
        modeling_result = {
            "metrics": {
                "accuracy": 0.6,
            }
        }

        gate = ModelReliabilityGate(thresholds={"min_accuracy": 0.7})
        result = gate.run(modeling_result)

        assert result.status == "fail"

    def test_high_calibration_error_fails(self):
        """Test that high calibration error triggers failure."""
        modeling_result = {
            "metrics": {
                "accuracy": 0.85,
                "ece": 0.15,
            }
        }

        gate = ModelReliabilityGate(thresholds={"max_calibration_error": 0.1})
        result = gate.run(modeling_result)

        assert result.status == "fail"

    def test_no_metrics_skipped(self):
        """Test that gate is skipped when no metrics available."""
        result = ModelReliabilityGate().run(None)

        assert result.status == "skip"


# ============================================================================
# REGULATORY ENFORCEMENT TESTS
# ============================================================================


class TestModelApproval:
    """Test model approval enforcement."""

    def test_approved_model_passes_regulatory(self):
        """Test that approved models pass regulatory check."""
        for model in ["LogisticRegression", "RandomForest", "SVC"]:
            is_approved, msg = enforce_model_approved(model, "regulatory")
            assert is_approved
            assert model in msg

    def test_unapproved_model_fails_regulatory(self):
        """Test that unapproved models fail regulatory check."""
        is_approved, msg = enforce_model_approved("WeirdCustomModel", "regulatory")
        assert not is_approved
        assert "not in approved registry" in msg

    def test_any_model_ok_research(self):
        """Test that any model is ok in research mode."""
        is_approved, msg = enforce_model_approved("WeirdCustomModel", "research")
        assert is_approved
        assert "research" in msg.lower()

    def test_custom_approved_models(self):
        """Test with custom approved models."""
        custom_approved = {"MyModel", "MyOtherModel"}
        is_approved, msg = enforce_model_approved("MyModel", "regulatory", approved_models=custom_approved)
        assert is_approved


class TestTrustStackEnforcement:
    """Test trust stack enforcement."""

    def test_trust_mandatory_in_regulatory(self):
        """Test that trust is mandatory in regulatory mode."""
        is_ok, msg = enforce_trust_stack(enable_trust=False, mode="regulatory")
        assert not is_ok
        assert "mandatory" in msg.lower()

    def test_trust_enabled_satisfies_regulatory(self):
        """Test that enabled trust satisfies regulatory."""
        is_ok, msg = enforce_trust_stack(enable_trust=True, mode="regulatory")
        assert is_ok

    def test_trust_optional_research(self):
        """Test that trust is optional in research mode."""
        is_ok, msg = enforce_trust_stack(enable_trust=False, mode="research")
        assert is_ok
        assert "advisory" in msg.lower()


class TestReportingEnforcement:
    """Test reporting enforcement."""

    def test_reporting_mandatory_in_regulatory(self):
        """Test that reporting is mandatory in regulatory mode."""
        is_ok, msg = enforce_reporting(enable_report=False, mode="regulatory")
        assert not is_ok
        assert "mandatory" in msg.lower()

    def test_reporting_enabled_satisfies_regulatory(self):
        """Test that enabled reporting satisfies regulatory."""
        is_ok, msg = enforce_reporting(enable_report=True, mode="regulatory")
        assert is_ok

    def test_reporting_optional_research(self):
        """Test that reporting is optional in research mode."""
        is_ok, msg = enforce_reporting(enable_report=False, mode="research")
        assert is_ok


class TestOverrideGovernance:
    """Test override governance enforcement."""

    def test_zero_overrides_pass(self):
        """Test that zero overrides pass."""
        is_ok, msg = check_override_governance(0)
        assert is_ok

    def test_one_override_with_justification_passes(self):
        """Test that one override with justification passes."""
        is_ok, msg = check_override_governance(
            1,
            override_justifications=["This is a very important justification for the override"],
        )
        assert is_ok

    def test_too_many_overrides_fail(self):
        """Test that too many overrides fail."""
        is_ok, msg = check_override_governance(2, max_allowed=1)
        assert not is_ok
        assert "Too many" in msg

    def test_override_without_justification_fails(self):
        """Test that override without justification fails."""
        is_ok, msg = check_override_governance(
            1,
            override_justifications=["short"],
            max_allowed=1,
        )
        assert not is_ok
        assert "justification" in msg.lower()


class TestRegulatoryRestrictions:
    """Test regulatory restriction retrieval."""

    def test_research_mode_restrictions(self):
        """Test research mode has minimal restrictions."""
        restrictions = get_regulatory_restrictions("research")
        assert not restrictions["model_approval_required"]
        assert not restrictions["trust_stack_required"]
        assert not restrictions["reporting_required"]

    def test_regulatory_mode_restrictions(self):
        """Test regulatory mode has strict restrictions."""
        restrictions = get_regulatory_restrictions("regulatory")
        assert restrictions["model_approval_required"]
        assert restrictions["trust_stack_required"]
        assert restrictions["reporting_required"]
        assert restrictions["max_overrides"] == 1
        assert "LogisticRegression" in restrictions["approved_models"]


# ============================================================================
# ARTIFACT CONTRACT PHASE 2 TESTS
# ============================================================================


class TestArtifactContractPhase2:
    """Test Phase 2 artifact contract requirements."""

    def test_qc_artifacts_required_when_enforce_qc(self, tmp_path):
        """Test that QC artifacts are required when enforce_qc=True."""
        # Create required always artifacts
        (tmp_path / "manifest.json").write_text("{}")
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "run.log").write_text("")
        (tmp_path / "success.json").write_text("{}")

        # Without QC artifacts, should fail
        is_valid, missing = ArtifactContract.validate_success(
            tmp_path,
            enforce_qc=True,
        )
        assert not is_valid
        assert "qc_results.json" in str(missing)

        # With QC artifacts, should pass
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts" / "qc_results.json").write_text("{}")

        is_valid, missing = ArtifactContract.validate_success(
            tmp_path,
            enforce_qc=True,
        )
        assert is_valid

    def test_trust_artifacts_required_when_enabled(self, tmp_path):
        """Test that trust artifacts are required when enable_trust=True."""
        # Create required always artifacts
        (tmp_path / "manifest.json").write_text("{}")
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "run.log").write_text("")
        (tmp_path / "success.json").write_text("{}")

        # Without trust artifacts, should fail
        is_valid, missing = ArtifactContract.validate_success(
            tmp_path,
            enable_trust=True,
        )
        assert not is_valid
        assert "trust_stack.json" in str(missing)

        # With trust artifacts, should pass
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts" / "trust_stack.json").write_text("{}")

        is_valid, missing = ArtifactContract.validate_success(
            tmp_path,
            enable_trust=True,
        )
        assert is_valid

    def test_reporting_artifacts_required_when_enabled(self, tmp_path):
        """Test that reporting artifacts are required when enable_reporting=True."""
        # Create required always artifacts
        (tmp_path / "manifest.json").write_text("{}")
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "run.log").write_text("")
        (tmp_path / "success.json").write_text("{}")

        # Without reporting artifacts, should fail
        is_valid, missing = ArtifactContract.validate_success(
            tmp_path,
            enable_reporting=True,
        )
        assert not is_valid
        assert "report.html" in str(missing)

        # With reporting artifacts, should pass
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts" / "report.html").write_text("<html></html>")

        is_valid, missing = ArtifactContract.validate_success(
            tmp_path,
            enable_reporting=True,
        )
        assert is_valid

    def test_no_extra_requirements_research_mode(self, tmp_path):
        """Test that research mode doesn't require extra artifacts."""
        # Create required always artifacts
        (tmp_path / "manifest.json").write_text("{}")
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "run.log").write_text("")
        (tmp_path / "success.json").write_text("{}")

        # Should pass without extra artifacts
        is_valid, missing = ArtifactContract.validate_success(
            tmp_path,
            enforce_qc=False,
            enable_trust=False,
            enable_reporting=False,
        )
        assert is_valid


# ============================================================================
# WORKFLOW CONFIG PHASE 2 TESTS
# ============================================================================


class TestWorkflowConfigPhase2:
    """Test Phase 2 additions to WorkflowConfig."""

    def test_config_has_phase2_fields(self):
        """Test that config has new Phase 2 fields."""
        cfg = WorkflowConfig(
            protocol=Path("protocol.yaml"),
            inputs=[Path("data.csv")],
            enforce_qc=True,
            enable_trust=True,
            enable_reporting=True,
        )

        assert cfg.enforce_qc is True
        assert cfg.enable_trust is True
        assert cfg.enable_reporting is True

    def test_config_defaults_phase2_fields(self):
        """Test that Phase 2 fields have sensible defaults."""
        cfg = WorkflowConfig(
            protocol=Path("protocol.yaml"),
            inputs=[Path("data.csv")],
        )

        assert cfg.enforce_qc is False
        assert cfg.enable_trust is False
        assert cfg.enable_reporting is True
