"""
Tests for trust artifact integration in the orchestrator.

Verifies that trust configuration is captured in the manifest
and artifact paths are recorded.

Note: These tests verify the manifest building and artifact path registration.
The actual implementation of trust feature execution in the orchestrator
(conformal prediction, calibration, etc.) is deferred to a future stage.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.manifest import RunManifest
from foodspec.core.orchestrator import ExecutionEngine
from foodspec.core.protocol import ProtocolV2


def test_trust_artifact_paths_registered_in_manifest() -> None:
    """Test that trust artifact paths are registered in manifest."""
    with TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "run_output"
        engine = ExecutionEngine(enable_cache=False)

        # Use minimal protocol without unimplemented features
        protocol = ProtocolV2(
            version="2.0.0",
            data={"input": "dummy.csv", "modality": "raman", "label": "label"},
            task={"name": "classification", "objective": "classify"},
        )

        result = engine.run(protocol, outdir, seed=42)

        # Verify all trust artifacts are registered in manifest
        # even though they won't be populated until trust stage is implemented
        trust_artifacts = [
            "calibration_metrics",
            "conformal_coverage",
            "conformal_sets",
            "abstention_summary",
            "coefficients",
            "permutation_importance",
            "marker_panel_explanations",
        ]

        for artifact_name in trust_artifacts:
            assert artifact_name in result.manifest.artifacts, \
                f"Trust artifact '{artifact_name}' should be registered in manifest"
            artifact_path = result.manifest.artifacts[artifact_name]
            assert artifact_path is not None
            assert len(artifact_path) > 0


def test_artifact_registry_trust_paths_exist() -> None:
    """Test that ArtifactRegistry provides all trust artifact paths."""
    with TemporaryDirectory() as tmpdir:
        artifacts = ArtifactRegistry(Path(tmpdir))

        # Verify all trust paths are accessible
        assert hasattr(artifacts, "calibration_metrics_path")
        assert hasattr(artifacts, "conformal_coverage_path")
        assert hasattr(artifacts, "conformal_sets_path")
        assert hasattr(artifacts, "abstention_summary_path")
        assert hasattr(artifacts, "coefficients_path")
        assert hasattr(artifacts, "permutation_importance_path")
        assert hasattr(artifacts, "marker_panel_explanations_path")

        # Verify paths are under trust directory
        assert "trust" in str(artifacts.calibration_metrics_path)
        assert "trust" in str(artifacts.conformal_coverage_path)
        assert "trust" in str(artifacts.conformal_sets_path)
        assert "trust" in str(artifacts.abstention_summary_path)


def test_manifest_build_with_trust_config() -> None:
    """Test that RunManifest.build correctly handles trust_config."""
    trust_config = {
        "conformal_enabled": True,
        "calibration_enabled": True,
        "abstention_enabled": False,
        "interpretability_enabled": True,
    }

    manifest = RunManifest.build(
        protocol_snapshot={"version": "2.0.0"},
        data_path=None,
        seed=42,
        artifacts={"test": "path"},
        trust_config=trust_config,
    )

    assert manifest.trust_config == trust_config


def test_manifest_build_with_empty_trust_config() -> None:
    """Test that RunManifest.build handles missing trust_config gracefully."""
    manifest = RunManifest.build(
        protocol_snapshot={"version": "2.0.0"},
        data_path=None,
        seed=42,
        artifacts={"test": "path"},
    )

    # Should have empty trust_config
    assert manifest.trust_config == {}


def test_trust_config_with_minimal_protocol() -> None:
    """Test that trust_config is built correctly with minimal protocol."""
    with TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "run_output"
        engine = ExecutionEngine(enable_cache=False)

        protocol = ProtocolV2(
            version="2.0.0",
            data={"input": "dummy.csv", "modality": "raman", "label": "label"},
            task={"name": "classification", "objective": "classify"},
        )

        result = engine.run(protocol, outdir, seed=42)

        # Verify trust_config is in manifest with all false
        assert "trust_config" in result.manifest.__dict__
        assert result.manifest.trust_config["conformal_enabled"] is False
        assert result.manifest.trust_config["calibration_enabled"] is False
        assert result.manifest.trust_config["abstention_enabled"] is False
        assert result.manifest.trust_config["interpretability_enabled"] is False


__all__ = [
    "test_trust_artifact_paths_registered_in_manifest",
    "test_artifact_registry_trust_paths_exist",
    "test_manifest_build_with_trust_config",
    "test_manifest_build_with_empty_trust_config",
    "test_trust_config_with_minimal_protocol",
]
