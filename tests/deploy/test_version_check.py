"""Tests for artifact version compatibility checking."""

from __future__ import annotations

import pytest

from foodspec.deploy.version_check import (
    CompatibilityLevel,
    VersionCompatibilityReport,
    check_version_compatibility,
    parse_semver,
    validate_artifact_compatibility,
)


class TestParseSemver:
    """Tests for semantic version parsing."""

    def test_parse_standard_version(self):
        """Test parsing standard semver strings."""
        assert parse_semver("1.2.3") == (1, 2, 3)
        assert parse_semver("0.9.1") == (0, 9, 1)
        assert parse_semver("10.20.30") == (10, 20, 30)

    def test_parse_version_with_prerelease(self):
        """Test parsing versions with pre-release tags."""
        assert parse_semver("1.2.3rc1") == (1, 2, 3)
        assert parse_semver("0.9.1alpha") == (0, 9, 1)
        assert parse_semver("2.0.0beta2") == (2, 0, 0)

    def test_parse_short_version(self):
        """Test parsing versions with missing components."""
        assert parse_semver("1.2") == (1, 2, 0)
        assert parse_semver("3") == (3, 0, 0)

    def test_parse_none_version(self):
        """Test parsing None returns (0, 0, 0)."""
        assert parse_semver(None) == (0, 0, 0)

    def test_parse_invalid_version(self):
        """Test parsing invalid strings returns (0, 0, 0)."""
        assert parse_semver("invalid") == (0, 0, 0)
        assert parse_semver("a.b.c") == (0, 0, 0)
        assert parse_semver("") == (0, 0, 0)


class TestCheckVersionCompatibility:
    """Tests for version compatibility checking."""

    def test_exact_match_compatible(self):
        """Test exact version match is fully compatible."""
        report = check_version_compatibility("1.2.3", "1.2.3")

        assert report.compatible
        assert report.level == CompatibilityLevel.COMPATIBLE
        assert len(report.errors) == 0
        assert len(report.warnings) == 0

    def test_patch_difference_compatible(self):
        """Test patch version differences are compatible."""
        report = check_version_compatibility("1.2.3", "1.2.5")

        assert report.compatible
        assert report.level == CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
        assert len(report.errors) == 0
        assert len(report.warnings) > 0
        assert "patch version difference" in report.warnings[0].lower()

    def test_older_minor_version_compatible(self):
        """Test loading older minor versions is compatible."""
        report = check_version_compatibility("1.2.0", "1.5.0")

        assert report.compatible
        assert report.level == CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
        assert len(report.errors) == 0
        assert "older minor version" in report.warnings[0].lower()

    def test_newer_minor_version_incompatible(self):
        """Test loading newer minor versions is incompatible."""
        report = check_version_compatibility("1.5.0", "1.2.0")

        assert not report.compatible
        assert report.level == CompatibilityLevel.INCOMPATIBLE
        assert len(report.errors) > 0
        assert "future minor version" in report.errors[0].lower()

    def test_newer_minor_allowed_with_flag(self):
        """Test allow_future_minor flag permits newer versions."""
        report = check_version_compatibility("1.5.0", "1.2.0", allow_future_minor=True)

        assert report.compatible
        assert report.level == CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
        assert len(report.warnings) > 0
        assert "future minor version" in report.warnings[0].lower()

    def test_major_version_mismatch_incompatible(self):
        """Test major version mismatch is always incompatible."""
        report = check_version_compatibility("2.0.0", "1.5.0")

        assert not report.compatible
        assert report.level == CompatibilityLevel.INCOMPATIBLE
        assert len(report.errors) > 0
        assert "major version mismatch" in report.errors[0].lower()

    def test_major_version_backward_incompatible(self):
        """Test loading older major version is incompatible."""
        report = check_version_compatibility("1.9.9", "2.0.0")

        assert not report.compatible
        assert report.level == CompatibilityLevel.INCOMPATIBLE
        assert "major version mismatch" in report.errors[0].lower()

    def test_missing_version_unknown(self):
        """Test missing version info returns UNKNOWN."""
        report = check_version_compatibility(None, "1.2.3")

        assert report.compatible  # Allows loading but warns
        assert report.level == CompatibilityLevel.UNKNOWN
        assert len(report.warnings) > 0
        assert "version information missing" in report.warnings[0].lower()

    def test_both_versions_missing_unknown(self):
        """Test both versions missing returns UNKNOWN."""
        report = check_version_compatibility(None, None)

        assert report.compatible
        assert report.level == CompatibilityLevel.UNKNOWN
        assert len(report.warnings) > 0

    def test_strict_mode_patch_difference(self):
        """Test strict mode rejects any version difference."""
        report = check_version_compatibility("1.2.3", "1.2.5", strict=True)

        assert not report.compatible
        assert report.level == CompatibilityLevel.INCOMPATIBLE
        assert "strict mode" in report.errors[0].lower()

    def test_strict_mode_minor_difference(self):
        """Test strict mode rejects minor version difference."""
        report = check_version_compatibility("1.2.0", "1.5.0", strict=True)

        assert not report.compatible
        assert "strict mode" in report.errors[0].lower()

    def test_recommendations_present(self):
        """Test recommendations are provided for incompatibility."""
        report = check_version_compatibility("2.0.0", "1.5.0")

        assert not report.compatible
        assert len(report.recommendations) > 0
        assert any("re-train" in rec.lower() for rec in report.recommendations)

    def test_report_str_representation(self):
        """Test string representation of compatibility report."""
        report = check_version_compatibility("1.2.3", "1.2.3")

        report_str = str(report)
        assert "COMPATIBLE" in report_str
        assert "1.2.3" in report_str

    def test_report_str_with_errors(self):
        """Test string representation includes errors."""
        report = check_version_compatibility("2.0.0", "1.5.0")

        report_str = str(report)
        assert "ERRORS" in report_str or "❌" in report_str
        assert "major version" in report_str.lower()

    def test_report_str_with_warnings(self):
        """Test string representation includes warnings."""
        report = check_version_compatibility("1.2.0", "1.5.0")

        report_str = str(report)
        assert "WARNINGS" in report_str or "⚠️" in report_str


class TestValidateArtifactCompatibility:
    """Tests for artifact validation with side effects."""

    def test_compatible_versions_no_error(self):
        """Test compatible versions don't raise errors."""
        report = validate_artifact_compatibility("1.2.3", "1.2.5", raise_on_incompatible=True)

        assert report.compatible

    def test_incompatible_raises_error(self):
        """Test incompatible versions raise ValueError."""
        with pytest.raises(ValueError, match="major version"):
            validate_artifact_compatibility("2.0.0", "1.5.0", raise_on_incompatible=True)

    def test_incompatible_no_raise_with_flag(self):
        """Test raise_on_incompatible=False doesn't raise."""
        with pytest.warns(UserWarning):
            report = validate_artifact_compatibility("2.0.0", "1.5.0", raise_on_incompatible=False)

        assert not report.compatible

    def test_warnings_emitted_for_minor_difference(self):
        """Test warnings are emitted for minor version differences."""
        with pytest.warns(UserWarning):
            validate_artifact_compatibility("1.2.0", "1.5.0")

    def test_strict_mode_raises_on_patch_difference(self):
        """Test strict mode raises even for patch differences."""
        with pytest.raises(ValueError, match="Strict mode"):
            validate_artifact_compatibility("1.2.3", "1.2.5", strict=True, raise_on_incompatible=True)


class TestVersionCompatibilityEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_major_version(self):
        """Test 0.x.x versions (development versions)."""
        report = check_version_compatibility("0.9.1", "0.9.5")

        assert report.compatible
        assert report.level == CompatibilityLevel.COMPATIBLE_WITH_WARNINGS

    def test_zero_to_one_major_incompatible(self):
        """Test transition from 0.x to 1.x is incompatible."""
        report = check_version_compatibility("0.9.9", "1.0.0")

        assert not report.compatible
        assert "major version" in report.errors[0].lower()

    def test_large_minor_version_gap(self):
        """Test large minor version differences."""
        report = check_version_compatibility("1.2.0", "1.20.0")

        assert report.compatible
        assert "older minor version" in report.warnings[0].lower()

    def test_version_with_build_metadata(self):
        """Test versions with build metadata are parsed correctly."""
        # Build metadata after '+' should be ignored in comparison
        report = check_version_compatibility("1.2.3+build123", "1.2.3+build456")

        # Both should parse to (1, 2, 3) and be considered compatible
        assert report.compatible

    def test_prerelease_versions(self):
        """Test pre-release versions are handled."""
        report = check_version_compatibility("1.2.3rc1", "1.2.3")

        # Both parse to (1, 2, 3), so should be compatible
        assert report.compatible


class TestVersionCompatibilityReportDataclass:
    """Test VersionCompatibilityReport dataclass."""

    def test_report_creation(self):
        """Test manual report creation."""
        report = VersionCompatibilityReport(
            level=CompatibilityLevel.COMPATIBLE,
            saved_version="1.2.3",
            current_version="1.2.3",
            artifact_schema_version="1.0",
            compatible=True,
            warnings=[],
            errors=[],
            recommendations=[],
        )

        assert report.compatible
        assert report.level == CompatibilityLevel.COMPATIBLE

    def test_report_with_warnings_and_errors(self):
        """Test report with both warnings and errors."""
        report = VersionCompatibilityReport(
            level=CompatibilityLevel.INCOMPATIBLE,
            saved_version="2.0.0",
            current_version="1.5.0",
            artifact_schema_version="1.0",
            compatible=False,
            warnings=["Some warning"],
            errors=["Major version mismatch"],
            recommendations=["Re-train model"],
        )

        assert not report.compatible
        assert len(report.warnings) == 1
        assert len(report.errors) == 1
        assert len(report.recommendations) == 1
