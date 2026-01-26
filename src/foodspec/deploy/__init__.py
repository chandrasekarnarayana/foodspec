"""Deployment and artifact management utilities."""

from foodspec.deploy.artifact import Predictor, load_artifact, save_artifact
from foodspec.deploy.export import export_onnx, export_pmml, load_pipeline, save_pipeline, PipelineBundle
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
    "PipelineBundle",
    "save_pipeline",
    "load_pipeline",
    "export_onnx",
    "export_pmml",
    # Version checking
    "CompatibilityLevel",
    "VersionCompatibilityReport",
    "check_version_compatibility",
    "parse_semver",
    "validate_artifact_compatibility",
]
