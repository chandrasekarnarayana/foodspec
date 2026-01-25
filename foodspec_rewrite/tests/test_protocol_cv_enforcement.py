"""
Comprehensive tests for "No random CV for real food data" enforcement.

Tests cover:
- Default scheme selection (LOBO when batch present)
- Random CV enforcement with error messages
- Override functionality with warnings
- Critical task handling (authentication/adulteration)
"""

import pytest

from foodspec.core.protocol import (
    DataSpec,
    ProtocolV2,
    TaskSpec,
    ValidationSpec,
)


# =============================================================================
# Test Default LOBO Selection
# =============================================================================


class TestDefaultLOBOSelection:
    """Test that LOBO is automatically selected when batch metadata present."""

    def test_auto_select_lobo_with_batch_key(self):
        """Should auto-select LOBO when batch key in metadata."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
        )
        
        # Before apply_defaults
        assert protocol.validation.scheme == "train_test_split"
        
        # Apply defaults should auto-upgrade to LOBO
        protocol_with_defaults = protocol.apply_defaults()
        
        assert protocol_with_defaults.validation.scheme == "leave_one_group_out"
        assert protocol_with_defaults.validation.group_key == "batch"

    def test_auto_select_lobo_with_stage_key(self):
        """Should auto-select LOBO when stage key in metadata."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "stage": "production_stage",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
        )
        
        protocol_with_defaults = protocol.apply_defaults()
        
        assert protocol_with_defaults.validation.scheme == "leave_one_group_out"
        assert protocol_with_defaults.validation.group_key == "stage"

    def test_auto_select_lobo_with_batch_id_key(self):
        """Should auto-select LOBO with batch_id variant."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch_id": "batch",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
        )
        
        protocol_with_defaults = protocol.apply_defaults()
        
        assert protocol_with_defaults.validation.scheme == "leave_one_group_out"
        assert protocol_with_defaults.validation.group_key == "batch_id"

    def test_no_auto_select_without_batch_keys(self):
        """Should NOT auto-select LOBO when no batch keys present."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
        )
        
        protocol_with_defaults = protocol.apply_defaults()
        
        # Should remain train_test_split
        assert protocol_with_defaults.validation.scheme == "train_test_split"
        assert protocol_with_defaults.validation.group_key is None

    def test_no_auto_select_if_scheme_already_set(self):
        """Should NOT override if user explicitly set a scheme."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="group_kfold", group_key="batch")
        )
        
        protocol_with_defaults = protocol.apply_defaults()
        
        # Should keep user's explicit choice
        assert protocol_with_defaults.validation.scheme == "group_kfold"


# =============================================================================
# Test Random CV Enforcement
# =============================================================================


class TestRandomCVEnforcement:
    """Test that random CV is blocked for food data with batch metadata."""

    def test_random_cv_raises_with_batch_metadata(self):
        """Random CV should raise error with batch metadata."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="quality_test", objective="classification"),
            validation=ValidationSpec(scheme="stratified_kfold")
        )
        
        with pytest.raises(ValueError, match="Random CV.*not recommended"):
            protocol.validate()

    def test_random_cv_raises_with_stage_metadata(self):
        """Random CV should raise error with stage metadata."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "stage": "ripening_stage",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="maturity_detection", objective="classification"),
            validation=ValidationSpec(scheme="kfold")
        )
        
        with pytest.raises(ValueError, match="Random CV.*not recommended"):
            protocol.validate()

    def test_random_cv_allowed_without_batch_metadata(self):
        """Random CV should work fine without batch metadata."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="stratified_kfold")
        )
        
        # Should NOT raise
        protocol.validate()


# =============================================================================
# Test Critical Task Enforcement
# =============================================================================


class TestCriticalTaskEnforcement:
    """Test strict enforcement for authentication/adulteration tasks."""

    def test_authentication_task_blocks_random_cv(self):
        """Authentication task should strictly block random CV with batch data."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="authentic",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "authentic",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="olive_oil_authentication", objective="authentication"),
            validation=ValidationSpec(scheme="stratified_kfold")
        )
        
        with pytest.raises(ValueError, match="Random CV.*is not allowed"):
            protocol.validate()
        
        # Error message should be detailed
        with pytest.raises(ValueError, match="critical for safety"):
            protocol.validate()

    def test_adulteration_task_blocks_random_cv(self):
        """Adulteration detection should strictly block random CV."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="nir",
                label="adulterated",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "adulterated",
                    "modality": "nir",
                }
            ),
            task=TaskSpec(name="adulteration_detection", objective="fraud detection"),
            validation=ValidationSpec(scheme="kfold")
        )
        
        with pytest.raises(ValueError, match="Random CV.*is not allowed"):
            protocol.validate()

    def test_fraud_task_blocks_random_cv(self):
        """Fraud detection should strictly block random CV."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="fraud",
                metadata_map={
                    "sample_id": "sample_id",
                    "stage": "production_stage",
                    "label": "fraud",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="fraud_detection", objective="classification"),
            validation=ValidationSpec(scheme="stratified_kfold")
        )
        
        with pytest.raises(ValueError, match="Random CV.*is not allowed"):
            protocol.validate()

    def test_origin_task_blocks_random_cv(self):
        """Origin verification should strictly block random CV."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="origin",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "origin",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="origin_verification", objective="geographic origin"),
            validation=ValidationSpec(scheme="kfold")
        )
        
        with pytest.raises(ValueError, match="Random CV.*is not allowed"):
            protocol.validate()


# =============================================================================
# Test Override with Warnings
# =============================================================================


class TestOverrideWithWarnings:
    """Test allow_random_cv override with warning recording."""

    def test_override_works_with_batch_metadata(self):
        """Override should allow random CV but record warning."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(
                scheme="stratified_kfold",
                allow_random_cv=True
            )
        )
        
        # Should NOT raise
        protocol.validate()
        
        # Should have warning
        assert len(protocol.validation.validation_warnings) == 1
        warning = protocol.validation.validation_warnings[0]
        assert "WARNING" in warning
        assert "Random CV override active" in warning
        assert "overestimated" in warning

    def test_override_works_for_critical_task(self):
        """Override should work even for critical tasks but with strong warning."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="authentic",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "authentic",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="authentication", objective="authentication"),
            validation=ValidationSpec(
                scheme="stratified_kfold",
                allow_random_cv=True
            )
        )
        
        # Should NOT raise
        protocol.validate()
        
        # Should have critical warning
        assert len(protocol.validation.validation_warnings) == 1
        warning = protocol.validation.validation_warnings[0]
        assert "WARNING" in warning
        assert "critical task" in warning
        assert "batch effects" in warning

    def test_warning_not_duplicated(self):
        """Warning should only be recorded once."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(
                scheme="stratified_kfold",
                allow_random_cv=True
            )
        )
        
        # Validate multiple times
        protocol.validate()
        protocol.validate()
        protocol.validate()
        
        # Should still have only one warning
        assert len(protocol.validation.validation_warnings) == 1


# =============================================================================
# Test Error Messages
# =============================================================================


class TestErrorMessages:
    """Test that error messages are actionable and informative."""

    def test_error_message_suggests_lobo(self):
        """Error message should suggest LOBO as solution."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="kfold")
        )
        
        with pytest.raises(ValueError, match="leave_one_group_out"):
            protocol.validate()

    def test_error_message_mentions_allow_random_cv(self):
        """Error message should mention override flag."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="stratified_kfold")
        )
        
        with pytest.raises(ValueError, match="allow_random_cv"):
            protocol.validate()

    def test_critical_task_error_explains_risks(self):
        """Critical task error should explain risks clearly."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="authentic",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "authentic",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="authentication", objective="authentication"),
            validation=ValidationSpec(scheme="kfold")
        )
        
        with pytest.raises(ValueError) as excinfo:
            protocol.validate()
        
        error_msg = str(excinfo.value)
        # Should explain risks
        assert "batch effects" in error_msg
        assert "overestimate performance" in error_msg
        # Should suggest actions
        assert "Recommended" in error_msg
        # Should explain override
        assert "domain experts" in error_msg or "understand the risks" in error_msg


# =============================================================================
# Test Integration
# =============================================================================


class TestProtocolIntegration:
    """Test integration with full protocol workflow."""

    def test_load_apply_validate_workflow(self):
        """Test full workflow with defaults and validation."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
        )
        
        # Apply defaults should auto-select LOBO
        protocol_with_defaults = protocol.apply_defaults()
        
        # Validate should pass
        protocol_with_defaults.validate()
        
        # Scheme should be LOBO
        assert protocol_with_defaults.validation.scheme == "leave_one_group_out"
        assert protocol_with_defaults.validation.group_key == "batch"

    def test_explicit_random_cv_requires_override(self):
        """User explicitly setting random CV should be caught."""
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={
                    "sample_id": "sample_id",
                    "batch": "batch_id",
                    "label": "target",
                    "modality": "raman",
                }
            ),
            task=TaskSpec(name="test", objective="classification"),
            validation=ValidationSpec(scheme="stratified_kfold")  # Explicit random CV
        )
        
        protocol_with_defaults = protocol.apply_defaults()
        
        # Should raise on validate
        with pytest.raises(ValueError, match="Random CV"):
            protocol_with_defaults.validate()
