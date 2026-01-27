"""Enhanced artifact version compatibility checking and validation.

This module provides comprehensive version compatibility checking for .foodspec
artifacts, ensuring safe deployment and preventing runtime errors from version
mismatches.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class CompatibilityLevel(Enum):
    """Compatibility level between artifact and runtime versions."""

    COMPATIBLE = "compatible"  # Fully compatible, no warnings
    COMPATIBLE_WITH_WARNINGS = "compatible_with_warnings"  # Works but may have issues
    INCOMPATIBLE = "incompatible"  # Cannot be safely loaded
    UNKNOWN = "unknown"  # Version info missing


@dataclass
class VersionCompatibilityReport:
    """Report on artifact-runtime version compatibility.

    Attributes
    ----------
    level : CompatibilityLevel
        Overall compatibility assessment.
    saved_version : str | None
        FoodSpec version used to create the artifact.
    current_version : str | None
        Current FoodSpec runtime version.
    artifact_schema_version : str | None
        Schema version of the artifact format.
    compatible : bool
        True if artifact can be loaded safely.
    warnings : list[str]
        Warning messages (minor version differences, deprecations).
    errors : list[str]
        Error messages (major version conflicts, missing dependencies).
    recommendations : list[str]
        Suggested actions for the user.
    """

    level: CompatibilityLevel
    saved_version: Optional[str]
    current_version: Optional[str]
    artifact_schema_version: Optional[str]
    compatible: bool
    warnings: list[str]
    errors: list[str]
    recommendations: list[str]

    def __str__(self) -> str:
        """Human-readable compatibility report."""
        lines = [
            f"Artifact Compatibility: {self.level.value.upper()}",
            f"  Saved Version: {self.saved_version or 'unknown'}",
            f"  Current Version: {self.current_version or 'unknown'}",
            f"  Schema Version: {self.artifact_schema_version or 'unknown'}",
        ]

        if self.errors:
            lines.append("\n❌ ERRORS:")
            for err in self.errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append("\n⚠️  WARNINGS:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        if self.recommendations:
            lines.append("\n💡 RECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


def parse_semver(version_str: Optional[str]) -> Tuple[int, int, int]:
    """Parse semantic version string into (major, minor, patch).

    Parameters
    ----------
    version_str : str | None
        Version string like "1.2.3" or "0.9.1rc1".

    Returns
    -------
    tuple[int, int, int]
        (major, minor, patch) tuple. Returns (0, 0, 0) for invalid input.

    Examples
    --------
    >>> parse_semver("1.2.3")
    (1, 2, 3)
    >>> parse_semver("0.9.1rc1")
    (0, 9, 1)
    >>> parse_semver(None)
    (0, 0, 0)
    """
    if not version_str:
        return (0, 0, 0)

    try:
        # Remove pre-release suffixes (rc1, alpha, beta, etc.)
        clean_version = str(version_str).split("rc")[0].split("alpha")[0].split("beta")[0]
        parts = [int(p) for p in clean_version.split(".")[:3]]

        # Pad to 3 elements
        while len(parts) < 3:
            parts.append(0)

        return tuple(parts[:3])  # type: ignore
    except Exception:
        return (0, 0, 0)


def check_version_compatibility(
    saved_version: Optional[str],
    current_version: Optional[str],
    artifact_schema_version: Optional[str] = None,
    *,
    allow_future_minor: bool = False,
    strict: bool = False,
) -> VersionCompatibilityReport:
    """Check if artifact version is compatible with current runtime.

    Implements semantic versioning compatibility rules:
    - Major version must match (1.x.x != 2.x.x)
    - Minor version: current >= saved (1.5.x can load 1.3.x)
    - Patch version: any difference allowed

    Parameters
    ----------
    saved_version : str | None
        Version of FoodSpec used to create the artifact.
    current_version : str | None
        Current FoodSpec runtime version.
    artifact_schema_version : str | None, optional
        Version of the artifact schema format.
    allow_future_minor : bool, default=False
        If True, allow loading artifacts from future minor versions (e.g., 1.5.x on 1.3.x).
        Generally unsafe but useful for development.
    strict : bool, default=False
        If True, treat any version mismatch as incompatible.

    Returns
    -------
    VersionCompatibilityReport
        Detailed compatibility assessment with warnings and recommendations.

    Examples
    --------
    >>> report = check_version_compatibility("1.2.3", "1.2.5")
    >>> report.compatible
    True
    >>> report.level
    <CompatibilityLevel.COMPATIBLE: 'compatible'>

    >>> report = check_version_compatibility("2.0.0", "1.5.0")
    >>> report.compatible
    False
    >>> report.level
    <CompatibilityLevel.INCOMPATIBLE: 'incompatible'>
    """
    warnings_list: list[str] = []
    errors_list: list[str] = []
    recommendations: list[str] = []

    # Handle missing versions
    if saved_version is None or current_version is None:
        warnings_list.append("Version information missing from artifact metadata")
        recommendations.append("Re-train the model with current FoodSpec version")
        return VersionCompatibilityReport(
            level=CompatibilityLevel.UNKNOWN,
            saved_version=saved_version,
            current_version=current_version,
            artifact_schema_version=artifact_schema_version,
            compatible=True,  # Allow loading but warn
            warnings=warnings_list,
            errors=errors_list,
            recommendations=recommendations,
        )

    # Parse versions
    s_major, s_minor, s_patch = parse_semver(saved_version)
    c_major, c_minor, c_patch = parse_semver(current_version)

    # Check major version compatibility (MUST match)
    if s_major != c_major:
        errors_list.append(f"Major version mismatch: artifact={s_major}.x.x, runtime={c_major}.x.x")
        errors_list.append("Breaking API changes between major versions prevent safe loading")
        recommendations.append(f"Re-train model with FoodSpec {c_major}.x.x")
        recommendations.append(f"Or downgrade runtime to FoodSpec {s_major}.x.x")

        return VersionCompatibilityReport(
            level=CompatibilityLevel.INCOMPATIBLE,
            saved_version=saved_version,
            current_version=current_version,
            artifact_schema_version=artifact_schema_version,
            compatible=False,
            warnings=warnings_list,
            errors=errors_list,
            recommendations=recommendations,
        )

    # Check minor version compatibility
    if s_minor > c_minor:
        if not allow_future_minor:
            errors_list.append(f"Artifact from future minor version: {s_major}.{s_minor}.x > {c_major}.{c_minor}.x")
            errors_list.append("Runtime may be missing features required by the artifact")
            recommendations.append(f"Upgrade FoodSpec to >={saved_version}")

            return VersionCompatibilityReport(
                level=CompatibilityLevel.INCOMPATIBLE,
                saved_version=saved_version,
                current_version=current_version,
                artifact_schema_version=artifact_schema_version,
                compatible=False,
                warnings=warnings_list,
                errors=errors_list,
                recommendations=recommendations,
            )
        else:
            warnings_list.append(f"Loading artifact from future minor version ({saved_version} on {current_version})")
            warnings_list.append("allow_future_minor=True; some features may not work correctly")

    elif s_minor < c_minor:
        warnings_list.append(f"Artifact from older minor version: {saved_version} < {current_version}")
        warnings_list.append("Artifact uses older API; should work but may miss improvements")
        recommendations.append("Consider re-training model with current version for latest features")

    # Check patch version (informational only)
    if s_patch != c_patch and s_minor == c_minor:
        warnings_list.append(
            f"Patch version difference: {s_major}.{s_minor}.{s_patch} vs {c_major}.{c_minor}.{c_patch}"
        )
        warnings_list.append("Patch differences (bug fixes) should not affect compatibility")

    # Strict mode: any mismatch is incompatible
    if strict and (s_minor != c_minor or s_patch != c_patch):
        errors_list.append("Strict mode: version must match exactly")
        recommendations.append(f"Re-train model with exact version {current_version}")

        return VersionCompatibilityReport(
            level=CompatibilityLevel.INCOMPATIBLE,
            saved_version=saved_version,
            current_version=current_version,
            artifact_schema_version=artifact_schema_version,
            compatible=False,
            warnings=warnings_list,
            errors=errors_list,
            recommendations=recommendations,
        )

    # Determine compatibility level
    if not warnings_list:
        level = CompatibilityLevel.COMPATIBLE
    else:
        level = CompatibilityLevel.COMPATIBLE_WITH_WARNINGS
        recommendations.append("Monitor for unexpected behavior in production")

    return VersionCompatibilityReport(
        level=level,
        saved_version=saved_version,
        current_version=current_version,
        artifact_schema_version=artifact_schema_version,
        compatible=True,
        warnings=warnings_list,
        errors=errors_list,
        recommendations=recommendations,
    )


def validate_artifact_compatibility(
    saved_version: Optional[str],
    current_version: Optional[str],
    *,
    allow_future_minor: bool = False,
    strict: bool = False,
    raise_on_incompatible: bool = True,
) -> VersionCompatibilityReport:
    """Validate artifact compatibility and optionally raise errors.

    Convenience function that checks compatibility and emits warnings/errors.

    Parameters
    ----------
    saved_version : str | None
        Version of FoodSpec used to create the artifact.
    current_version : str | None
        Current FoodSpec runtime version.
    allow_future_minor : bool, default=False
        Allow loading artifacts from future minor versions.
    strict : bool, default=False
        Require exact version match.
    raise_on_incompatible : bool, default=True
        If True, raise ValueError on incompatibility.
        If False, emit warnings and return report.

    Returns
    -------
    VersionCompatibilityReport
        Compatibility assessment.

    Raises
    ------
    ValueError
        If artifact is incompatible and raise_on_incompatible=True.

    Examples
    --------
    >>> report = validate_artifact_compatibility("1.2.0", "1.3.0")
    >>> report.compatible
    True

    >>> # This would raise ValueError:
    >>> # validate_artifact_compatibility("2.0.0", "1.5.0")
    """
    report = check_version_compatibility(
        saved_version,
        current_version,
        allow_future_minor=allow_future_minor,
        strict=strict,
    )

    # Emit warnings
    for warning in report.warnings:
        warnings.warn(warning, UserWarning, stacklevel=2)

    # Handle incompatibility
    if not report.compatible:
        error_msg = f"\n{report}"
        if raise_on_incompatible:
            raise ValueError(error_msg)
        else:
            warnings.warn(error_msg, UserWarning, stacklevel=2)

    return report


__all__ = [
    "CompatibilityLevel",
    "VersionCompatibilityReport",
    "parse_semver",
    "check_version_compatibility",
    "validate_artifact_compatibility",
]
