"""
Tests for mandatory metadata schema enforcement.

Verifies:
- Required metadata keys are enforced at load time
- LOBO/LOSO validation schemes require group_key in metadata
- QC grouping requires group_by key in metadata
- Actionable error messages list missing and available keys
- infer_required_metadata_keys() works correctly
"""

import pandas as pd
import pytest
from pathlib import Path

from foodspec.core.protocol import (
    DataSpec,
    ProtocolV2,
    TaskSpec,
    ValidationSpec,
    QCSpec,
)
from foodspec.core.data import SpectraSet
from foodspec.io.readers import load_csv_spectra


class TestDataSpecRequiredMetadata:
    """Test DataSpec required_metadata_keys field."""

    def test_dataspec_accepts_required_keys(self):
        """DataSpec should accept required_metadata_keys."""
        spec = DataSpec(
            input="data.csv",
            modality="raman",
            label="target",
            required_metadata_keys=["batch", "instrument"],
        )
        assert spec.required_metadata_keys == ["batch", "instrument"]

    def test_dataspec_default_empty_required_keys(self):
        """DataSpec required_metadata_keys defaults to empty list."""
        spec = DataSpec(
            input="data.csv",
            modality="raman",
            label="target",
        )
        assert spec.required_metadata_keys == []


class TestProtocolInferRequiredKeys:
    """Test ProtocolV2.infer_required_metadata_keys()."""

    def test_infer_from_group_validation(self):
        """Infer group_key from validation scheme."""
        protocol = ProtocolV2(
            data=DataSpec(input="data.csv", modality="raman", label="target"),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="group_kfold", group_key="batch"),
        )
        required = protocol.infer_required_metadata_keys()
        assert "batch" in required

    def test_infer_from_qc_grouping(self):
        """Infer group_by from QC config."""
        protocol = ProtocolV2(
            data=DataSpec(input="data.csv", modality="raman", label="target"),
            task=TaskSpec(name="test", objective="classification"),
            qc=QCSpec(group_by="instrument"),
        )
        required = protocol.infer_required_metadata_keys()
        assert "instrument" in required

    def test_infer_multiple_keys(self):
        """Infer multiple keys from validation and QC."""
        protocol = ProtocolV2(
            data=DataSpec(input="data.csv", modality="raman", label="target"),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="group_kfold", group_key="batch"),
            qc=QCSpec(group_by="instrument"),
        )
        required = protocol.infer_required_metadata_keys()
        assert "batch" in required
        assert "instrument" in required

    def test_infer_deduplicates(self):
        """Infer deduplicates repeated keys."""
        protocol = ProtocolV2(
            data=DataSpec(input="data.csv", modality="raman", label="target"),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="group_kfold", group_key="batch"),
            qc=QCSpec(group_by="batch"),  # Same key
        )
        required = protocol.infer_required_metadata_keys()
        assert required.count("batch") == 1

    def test_infer_empty_when_no_requirements(self):
        """Infer returns empty when no grouping configured."""
        protocol = ProtocolV2(
            data=DataSpec(input="data.csv", modality="raman", label="target"),
            task=TaskSpec(name="test", objective="classification"),
        )
        required = protocol.infer_required_metadata_keys()
        assert required == []


class TestLoadCSVSpectraEnforcement:
    """Test load_csv_spectra enforces required metadata."""

    def test_load_csv_with_all_required_metadata(self, tmp_path):
        """Load succeeds when all required metadata present."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "batch,instrument,target,400,500\n"
            "A,Inst1,class1,1.0,2.0\n"
            "B,Inst2,class2,3.0,4.0\n"
        )

        spec = DataSpec(
            input=str(csv_path),
            modality="raman",
            label="target",
            required_metadata_keys=["batch", "instrument"],
        )

        ds = load_csv_spectra(csv_path, spec)
        assert "batch" in ds.metadata.columns
        assert "instrument" in ds.metadata.columns

    def test_load_csv_missing_required_metadata_fails(self, tmp_path):
        """Load fails with actionable error when required metadata missing."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "batch,target,400,500\n"
            "A,class1,1.0,2.0\n"
            "B,class2,3.0,4.0\n"
        )

        spec = DataSpec(
            input=str(csv_path),
            modality="raman",
            label="target",
            required_metadata_keys=["batch", "instrument"],  # instrument missing
        )

        with pytest.raises(ValueError, match="Missing required metadata columns: instrument"):
            load_csv_spectra(csv_path, spec)

    def test_load_csv_error_shows_available_columns(self, tmp_path):
        """Error message lists available columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "batch,replicate,target,400,500\n"
            "A,1,class1,1.0,2.0\n"
        )

        spec = DataSpec(
            input=str(csv_path),
            modality="raman",
            label="target",
            required_metadata_keys=["instrument"],  # Not in CSV
        )

        with pytest.raises(ValueError) as exc_info:
            load_csv_spectra(csv_path, spec)

        assert "Available columns:" in str(exc_info.value)
        assert "batch" in str(exc_info.value)
        assert "replicate" in str(exc_info.value)

    def test_load_csv_with_metadata_map(self, tmp_path):
        """Required keys work with metadata_map remapping."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "exp_batch,exp_instrument,exp_target,400,500\n"
            "A,Inst1,class1,1.0,2.0\n"
        )

        spec = DataSpec(
            input=str(csv_path),
            modality="raman",
            label="exp_target",
            metadata_map={
                "batch": "exp_batch",
                "instrument": "exp_instrument",
                "label": "exp_target",
            },
            required_metadata_keys=["batch", "instrument"],
        )

        ds = load_csv_spectra(csv_path, spec)
        # After renaming via metadata_map
        assert "batch" in ds.metadata.columns
        assert "instrument" in ds.metadata.columns


class TestSpectraSetValidateRequiredMetadata:
    """Test SpectraSet.validate_required_metadata()."""

    def test_validate_passes_when_all_present(self):
        """Validation passes when all required keys present."""
        ds = SpectraSet(
            X=[[1, 2], [3, 4]],
            x=[100, 200],
            metadata=pd.DataFrame({"batch": ["A", "B"], "instrument": ["I1", "I2"]}),
        )
        ds.validate_required_metadata(["batch", "instrument"])  # Should not raise

    def test_validate_fails_when_missing(self):
        """Validation fails with actionable error when keys missing."""
        ds = SpectraSet(
            X=[[1, 2], [3, 4]],
            x=[100, 200],
            metadata=pd.DataFrame({"batch": ["A", "B"]}),
        )

        with pytest.raises(ValueError, match="Missing required metadata keys: instrument"):
            ds.validate_required_metadata(["batch", "instrument"])

    def test_validate_error_shows_available(self):
        """Error message shows available metadata keys."""
        ds = SpectraSet(
            X=[[1, 2]],
            x=[100, 200],
            metadata=pd.DataFrame({"batch": ["A"], "replicate": [1]}),
        )

        with pytest.raises(ValueError) as exc_info:
            ds.validate_required_metadata(["instrument", "matrix"])

        assert "Available: batch, replicate" in str(exc_info.value)

    def test_validate_empty_required_keys(self):
        """Validation passes with empty required keys list."""
        ds = SpectraSet(
            X=[[1, 2]],
            x=[100, 200],
            metadata=pd.DataFrame({"batch": ["A"]}),
        )
        ds.validate_required_metadata([])  # Should not raise


class TestWorkflowIntegration:
    """Test full workflow integration scenarios."""

    def test_lobo_validation_requires_batch(self, tmp_path):
        """LOBO validation should fail if batch column missing."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "target,400,500\n"
            "class1,1.0,2.0\n"
            "class2,3.0,4.0\n"
        )

        protocol = ProtocolV2(
            data=DataSpec(
                input=str(csv_path),
                modality="raman",
                label="target",
                required_metadata_keys=["batch"],  # Inferred or explicit
            ),
            task=TaskSpec(name="lobo", objective="classification"),
            validation=ValidationSpec(scheme="group_kfold", group_key="batch"),
        )

        with pytest.raises(ValueError, match="Missing required metadata columns: batch"):
            load_csv_spectra(csv_path, protocol.data)

    def test_qc_grouping_requires_metadata(self, tmp_path):
        """QC grouping should fail if group_by column missing."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "batch,target,400,500\n"
            "A,class1,1.0,2.0\n"
        )

        protocol = ProtocolV2(
            data=DataSpec(
                input=str(csv_path),
                modality="raman",
                label="target",
                required_metadata_keys=["instrument"],  # QC wants this
            ),
            task=TaskSpec(name="qc_test", objective="classification"),
            qc=QCSpec(group_by="instrument"),
        )

        with pytest.raises(ValueError, match="Missing required metadata columns: instrument"):
            load_csv_spectra(csv_path, protocol.data)

    def test_workflow_succeeds_with_all_metadata(self, tmp_path):
        """Workflow succeeds when all required metadata present."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "batch,instrument,target,400,500\n"
            "A,Inst1,class1,1.0,2.0\n"
            "A,Inst1,class1,1.5,2.5\n"
            "B,Inst2,class2,3.0,4.0\n"
        )

        # Infer required keys
        protocol = ProtocolV2(
            data=DataSpec(input=str(csv_path), modality="raman", label="target"),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="group_kfold", group_key="batch"),
            qc=QCSpec(group_by="instrument"),
        )

        required = protocol.infer_required_metadata_keys()
        protocol.data.required_metadata_keys = required

        # Should load successfully
        ds = load_csv_spectra(csv_path, protocol.data)
        assert "batch" in ds.metadata.columns
        assert "instrument" in ds.metadata.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
