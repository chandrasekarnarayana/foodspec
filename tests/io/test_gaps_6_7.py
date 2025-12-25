"""
Tests for gap 6 (OPUS/SPC vendor format support) and gap 7 (HDF5 schema versioning).
"""

import tempfile

import h5py
import numpy as np
import pytest

from foodspec.io.hdf5_schema_versioning import (
    CURRENT_SCHEMA_VERSION,
    CompatibilityLevel,
    SchemaVersion,
    check_schema_compatibility,
    get_compatibility_level,
    migrate_schema,
    migrate_schema_v1_0_to_v1_1,
    migrate_schema_v1_1_to_v1_2,
    migrate_schema_v1_2_to_v2_0,
)
from foodspec.io.vendor_format_support import (
    OPUS_BLOCK_TYPES_SUPPORTED,
    SPC_BLOCK_TYPES_SUPPORTED,
    BlockTypeSupportEntry,
    get_opus_support_summary,
    get_spc_support_summary,
    get_untested_blocks_opus,
    get_untested_blocks_spc,
    validate_opus_blocks,
    validate_spc_blocks,
)

# ============================================================================
# GAP 6: OPUS/SPC Vendor Format Support Tests
# ============================================================================

class TestOPUSBlockTypeSupport:
    """Test OPUS block type support matrix."""

    def test_opus_block_type_structure(self):
        """Verify all OPUS block types have proper metadata."""
        for block_type, entry in OPUS_BLOCK_TYPES_SUPPORTED.items():
            assert isinstance(entry, BlockTypeSupportEntry)
            assert entry.block_type == block_type
            assert entry.description is not None
            assert isinstance(entry.supported, bool)
            assert isinstance(entry.tested, bool)

    def test_opus_supported_blocks_exist(self):
        """Check that at least some OPUS blocks are supported."""
        supported = [k for k, v in OPUS_BLOCK_TYPES_SUPPORTED.items() if v.supported]
        assert len(supported) > 0
        assert "AB" in supported  # Main spectrum should be supported
        assert "DX" in supported  # Data should be supported

    def test_opus_tested_blocks_documented(self):
        """Verify tested blocks have min_version documented."""
        for block_type, entry in OPUS_BLOCK_TYPES_SUPPORTED.items():
            if entry.tested and entry.supported:
                # Tested blocks should have version info
                assert entry.min_version is not None

    def test_opus_untested_blocks_have_limitations(self):
        """Verify untested blocks document why."""
        for block_type, entry in OPUS_BLOCK_TYPES_SUPPORTED.items():
            if entry.supported and not entry.tested:
                # Untested blocks should explain why
                assert entry.known_limitations is not None

    def test_opus_block_summary_readable(self):
        """Test that support summary is human-readable."""
        summary = get_opus_support_summary()
        assert "OPUS Block Type Support Summary" in summary
        assert "Fully Supported & Tested" in summary
        assert "Supported but Untested" in summary
        assert "Unsupported" in summary
        assert len(summary) > 100  # Should be substantive


class TestSPCBlockTypeSupport:
    """Test SPC block type support matrix."""

    def test_spc_block_type_structure(self):
        """Verify all SPC block types have proper metadata."""
        for block_type, entry in SPC_BLOCK_TYPES_SUPPORTED.items():
            assert isinstance(entry, BlockTypeSupportEntry)
            assert entry.description is not None
            assert isinstance(entry.supported, bool)

    def test_spc_supported_blocks_exist(self):
        """Check that core SPC blocks are supported."""
        supported = [k for k, v in SPC_BLOCK_TYPES_SUPPORTED.items() if v.supported]
        assert "data" in supported
        assert "x_axis" in supported
        assert len(supported) > 0

    def test_spc_block_summary_readable(self):
        """Test that SPC support summary is human-readable."""
        summary = get_spc_support_summary()
        assert "SPC Block Type Support Summary" in summary
        assert len(summary) > 50


class TestVendorBlockValidation:
    """Test block validation functions."""

    def test_validate_opus_blocks_all_supported(self):
        """Test validation of all-supported OPUS blocks."""
        blocks = {"AB", "DX", "PA"}
        validation = validate_opus_blocks(blocks)

        assert validation["AB"] is True
        assert validation["DX"] is True
        assert validation["PA"] is True

    def test_validate_opus_blocks_mixed_support(self):
        """Test validation with mixed support."""
        blocks = {"AB", "GX", "PE"}  # GX and PE are unsupported
        validation = validate_opus_blocks(blocks)

        assert validation["AB"] is True
        assert validation["GX"] is False
        assert validation["PE"] is False

    def test_validate_spc_blocks_all_supported(self):
        """Test validation of all-supported SPC blocks."""
        blocks = {"data", "x_axis", "log_data"}
        validation = validate_spc_blocks(blocks)

        assert all(validation.values())

    def test_validate_spc_blocks_with_unsupported(self):
        """Test SPC validation with unsupported blocks."""
        blocks = {"data", "interferogram"}
        validation = validate_spc_blocks(blocks)

        assert validation["data"] is True
        assert validation["interferogram"] is False

    def test_get_untested_opus_blocks(self):
        """Test retrieval of untested OPUS blocks."""
        blocks = {"HX", "OP", "RX"}  # Mix of tested/untested
        untested = get_untested_blocks_opus(blocks)

        # Should include HX, OP, RX (supported but untested)
        assert len(untested) > 0
        assert all(
            OPUS_BLOCK_TYPES_SUPPORTED[b].supported for b in untested
        )
        assert all(
            not OPUS_BLOCK_TYPES_SUPPORTED[b].tested for b in untested
        )

    def test_get_untested_spc_blocks(self):
        """Test retrieval of untested SPC blocks."""
        blocks = {"sample_info", "data"}
        untested = get_untested_blocks_spc(blocks)

        # sample_info is supported but untested
        assert "sample_info" in untested


# ============================================================================
# GAP 7: HDF5 Schema Versioning Tests
# ============================================================================

class TestHDF5SchemaVersioning:
    """Test HDF5 schema versioning and compatibility."""

    def test_schema_version_enum(self):
        """Test schema version enum values."""
        assert SchemaVersion.V1_0.value == "1.0"
        assert SchemaVersion.V1_1.value == "1.1"
        assert SchemaVersion.V1_2.value == "1_2"
        assert SchemaVersion.V2_0.value == "2.0"

    def test_current_schema_version_defined(self):
        """Test that current schema version is valid."""
        assert CURRENT_SCHEMA_VERSION in ["1.0", "1.1", "1_2", "1.2", "2.0"]

    def test_compatibility_level_enum(self):
        """Test compatibility level enum."""
        levels = [
            CompatibilityLevel.COMPATIBLE,
            CompatibilityLevel.READABLE,
            CompatibilityLevel.INCOMPATIBLE,
            CompatibilityLevel.REQUIRES_MIGRATION,
        ]
        assert len(levels) == 4


class TestSchemaCompatibility:
    """Test schema compatibility checking."""

    def test_same_version_compatible(self):
        """Test that same versions are compatible."""
        assert (
            get_compatibility_level("1.0", "1.0")
            == CompatibilityLevel.COMPATIBLE
        )
        assert (
            get_compatibility_level("1.2", "1.2")
            == CompatibilityLevel.COMPATIBLE
        )
        assert (
            get_compatibility_level("2.0", "2.0")
            == CompatibilityLevel.COMPATIBLE
        )

    def test_older_versions_readable(self):
        """Test that older stable versions are readable."""
        assert (
            get_compatibility_level("1.0", "1.1")
            == CompatibilityLevel.READABLE
        )
        assert (
            get_compatibility_level("1.1", "1.2")
            == CompatibilityLevel.READABLE
        )

    def test_newer_versions_incompatible(self):
        """Test that newer file versions are incompatible."""
        assert (
            get_compatibility_level("1.1", "1.0")
            == CompatibilityLevel.INCOMPATIBLE
        )
        assert (
            get_compatibility_level("2.0", "1.2")
            == CompatibilityLevel.INCOMPATIBLE
        )

    def test_major_version_jump_requires_migration(self):
        """Test that major version jumps require migration."""
        assert (
            get_compatibility_level("1.0", "2.0")
            == CompatibilityLevel.REQUIRES_MIGRATION
        )

    def test_check_compatibility_compatible(self):
        """Test compatibility check for compatible versions."""
        assert check_schema_compatibility("1.0", "1.0") is True
        assert check_schema_compatibility("1.2", "1.2") is True

    def test_check_compatibility_readable_warns(self):
        """Test that readable versions issue warnings."""
        with pytest.warns(DeprecationWarning):
            result = check_schema_compatibility("1.0", "1.1")
            assert result is True

    def test_check_compatibility_migration_required_fails(self):
        """Test that migration-required versions fail by default."""
        with pytest.raises(RuntimeError):
            check_schema_compatibility("1.0", "1.2")

    def test_check_compatibility_migration_with_flag(self):
        """Test that migration-required versions work with flag."""
        with pytest.warns(UserWarning):
            result = check_schema_compatibility(
                "1.0", "1.2", allow_incompatible=True
            )
            assert result is True

    def test_check_compatibility_incompatible_fails(self):
        """Test that incompatible versions fail."""
        with pytest.raises(RuntimeError):
            check_schema_compatibility("2.0", "1.0")

    def test_check_compatibility_incompatible_with_flag(self):
        """Test that incompatible versions warn with flag."""
        with pytest.warns(UserWarning):
            result = check_schema_compatibility(
                "2.0", "1.0", allow_incompatible=True
            )
            assert result is True


class TestSchemaMigration:
    """Test HDF5 schema migration."""

    def test_migrate_v1_0_to_v1_1(self):
        """Test migration from v1.0 to v1.1."""
        with tempfile.TemporaryFile() as tmp:
            with h5py.File(tmp, "w") as f:
                group = f.create_group("test")
                group.attrs["schema_version"] = "1.0"

                migrate_schema_v1_0_to_v1_1(group)

                # Check preprocessing_history was added
                assert "preprocessing_history" in group

    def test_migrate_v1_1_to_v1_2(self):
        """Test migration from v1.1 to v1.2."""
        with tempfile.TemporaryFile() as tmp:
            with h5py.File(tmp, "w") as f:
                group = f.create_group("test")
                group.attrs["schema_version"] = "1.1"

                migrate_schema_v1_1_to_v1_2(group)

                # Check artifact versioning was added
                assert "artifact_version" in group.attrs
                assert "foodspec_version" in group.attrs

    def test_migrate_v1_2_to_v2_0(self):
        """Test migration from v1.2 to v2.0."""
        with tempfile.TemporaryFile() as tmp:
            with h5py.File(tmp, "w") as f:
                group = f.create_group("test")
                group.attrs["schema_version"] = "1.2"

                migrate_schema_v1_2_to_v2_0(group)

                # Check streaming metadata was added
                assert "streaming_capable" in group.attrs
                assert bool(group.attrs["streaming_capable"]) is True

    def test_full_migration_v1_0_to_v1_2(self):
        """Test full migration path from v1.0 to v1.2."""
        with tempfile.TemporaryFile() as tmp:
            with h5py.File(tmp, "w") as f:
                group = f.create_group("test")
                group.attrs["schema_version"] = "1.0"

                migrate_schema(group, "1.0", "1.2")

                # Check all expected attributes/groups
                assert "preprocessing_history" in group
                assert "artifact_version" in group.attrs
                assert group.attrs["schema_version"] == "1.2"

    def test_migration_preserves_data(self):
        """Test that migration preserves existing data."""
        with tempfile.TemporaryFile() as tmp:
            with h5py.File(tmp, "w") as f:
                group = f.create_group("test")
                group.attrs["schema_version"] = "1.0"

                # Add sample data
                test_data = np.random.randn(10, 100)
                group.create_dataset("spectra", data=test_data)
                group.attrs["sample_name"] = "test_sample"

                migrate_schema(group, "1.0", "1.2")

                # Verify original data preserved
                assert "spectra" in group
                assert np.allclose(group["spectra"][:], test_data)
                assert group.attrs["sample_name"] == "test_sample"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
