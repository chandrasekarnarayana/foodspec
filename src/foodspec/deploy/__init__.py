"""Deployment and artifact management utilities."""

from foodspec.deploy.artifact import Predictor, load_artifact, save_artifact
from foodspec.deploy.version_check import (
    CompatibilityLevel,
    VersionCompatibilityReport,
    check_version_compatibility,
    parse_semver,
    validate_artifact_compatibility,
)

__all__ = [
    # Artifact management
    "save_artifact",
    "load_artifact",
    "Predictor",
    # Version checking
    "CompatibilityLevel",
    "VersionCompatibilityReport",
    "check_version_compatibility",
    "parse_semver",
    "validate_artifact_compatibility",
]
