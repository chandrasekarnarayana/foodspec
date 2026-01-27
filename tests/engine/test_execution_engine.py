"""
Comprehensive tests for FoodSpec Execution Engine.

Tests cover:
- Design Philosophy enforcement
- Pipeline DAG execution
- Artifact Registry tracking
- Run Manifest generation
- Deterministic execution
- End-to-end orchestration
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from foodspec.core.philosophy import (
    PhilosophyError,
    enforce_qc_first,
    enforce_report_first,
    enforce_reproducibility,
    enforce_task_first,
    enforce_trust_first,
    validate_all_principles,
)
from foodspec.core.run_manifest import (
    DataSnapshot,
    EnvironmentSnapshot,
    ManifestBuilder,
    ProtocolSnapshot,
)
from foodspec.engine.artifacts import (
    ArtifactRegistry,
    ArtifactType,
)
from foodspec.engine.dag import (
    NodeType,
    PipelineDAG,
    build_standard_pipeline,
)
from foodspec.engine.orchestrator import ExecutionEngine
from foodspec.utils.determinism import (
    capture_environment,
    capture_versions,
    fingerprint_csv,
    fingerprint_protocol,
    generate_reproducibility_report,
    get_global_seed,
    set_global_seed,
)

# ============================================================================
# Philosophy Enforcement Tests
# ============================================================================

class TestPhilosophyEnforcement:
    """Test design principle enforcement"""

    def test_task_first_valid(self):
        """Task in TASK_FIRST should pass"""
        config = {"task": "authentication"}
        enforce_task_first(config)  # Should not raise

    def test_task_first_invalid(self):
        """Task not in TASK_FIRST should fail"""
        config = {"task": "invalid_task"}
        with pytest.raises(PhilosophyError):
            enforce_task_first(config)

    def test_task_first_missing(self):
        """Missing task should fail"""
        config = {}
        with pytest.raises(PhilosophyError):
            enforce_task_first(config)

    def test_qc_first_valid(self):
        """Valid QC results should pass"""
        qc_results = {"status": "pass", "pass_rate": 0.95}
        enforce_qc_first(qc_results)  # Should not raise

    def test_qc_first_low_pass_rate(self):
        """QC pass rate below 50% should fail"""
        qc_results = {"status": "pass", "pass_rate": 0.3}
        with pytest.raises(PhilosophyError):
            enforce_qc_first(qc_results)

    def test_qc_first_critical_failure(self):
        """QC with critical failures should fail"""
        qc_results = {"status": "pass", "critical_failures": ["check_1"], "pass_rate": 0.8}
        with pytest.raises(PhilosophyError):
            enforce_qc_first(qc_results)

    def test_trust_first_valid(self):
        """Trust outputs with required keys should pass"""
        trust_outputs = {"calibration": {}, "conformal": {}}
        enforce_trust_first(trust_outputs)  # Should not raise

    def test_trust_first_missing_components(self):
        """Trust outputs missing components should fail"""
        trust_outputs = {"calibration": {}}
        with pytest.raises(PhilosophyError):
            enforce_trust_first(trust_outputs)

    def test_reproducibility_valid(self):
        """Complete reproducibility metadata should pass"""
        manifest = {
            "seed": 42,
            "python_version": "3.9.0",
            "os_info": "Linux",
            "package_versions": {},
            "data_fingerprint": "abc123",
        }
        enforce_reproducibility(manifest)  # Should not raise

    def test_reproducibility_missing_field(self):
        """Missing reproducibility field should fail"""
        manifest = {"seed": 42}  # Missing other fields
        with pytest.raises(PhilosophyError):
            enforce_reproducibility(manifest)

    def test_report_first_valid(self):
        """Valid report artifacts should pass"""
        artifacts = {"report_html": Path("/tmp/report.html")}
        enforce_report_first(artifacts)  # Should not raise

    def test_report_first_no_artifacts(self):
        """No report artifacts should fail"""
        artifacts = {}
        with pytest.raises(PhilosophyError):
            enforce_report_first(artifacts)


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterministicExecution:
    """Test deterministic execution infrastructure"""

    def test_global_seed_setting(self):
        """Setting global seed should update state"""
        set_global_seed(42)
        assert get_global_seed() == 42

        set_global_seed(100)
        assert get_global_seed() == 100

    def test_environment_capture(self):
        """Environment capture should return all fields"""
        env = capture_environment()

        assert "os" in env
        assert "python" in env
        assert "machine" in env
        assert "working_directory" in env

        assert env["os"]["name"] in ["Linux", "Windows", "Darwin"]
        assert "python_version" in env["python"]

    def test_version_capture(self):
        """Version capture should include critical packages"""
        versions = capture_versions()

        assert "critical_packages" in versions
        assert "numpy" in versions["critical_packages"]
        assert "pandas" in versions["critical_packages"]

        # Versions should not be empty strings
        for pkg, ver in versions["critical_packages"].items():
            assert ver and ver != ""

    def test_protocol_fingerprinting(self):
        """Protocol fingerprinting should be deterministic"""
        protocol = {"task": "authentication", "model": "svm"}

        hash1 = fingerprint_protocol(protocol)
        hash2 = fingerprint_protocol(protocol)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

    def test_protocol_fingerprinting_order(self):
        """Protocol fingerprinting should ignore key order"""
        proto_a = {"task": "auth", "model": "svm"}
        proto_b = {"model": "svm", "task": "auth"}

        assert fingerprint_protocol(proto_a) == fingerprint_protocol(proto_b)

    def test_csv_fingerprinting(self):
        """CSV fingerprinting should work"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,y\n1,2\n3,4\n")
            csv_path = Path(f.name)

        try:
            hash_val = fingerprint_csv(csv_path)
            assert len(hash_val) == 64  # SHA256

            # Same file should produce same hash
            hash_val2 = fingerprint_csv(csv_path)
            assert hash_val == hash_val2
        finally:
            csv_path.unlink()

    def test_reproducibility_report(self):
        """Reproducibility report generation should work"""
        protocol = {"task": "auth"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,y\n1,2\n")
            csv_path = Path(f.name)

        try:
            report = generate_reproducibility_report(
                seed=42,
                csv_path=csv_path,
                protocol_dict=protocol,
            )

            assert report.seed == 42
            assert report.data_fingerprint != "none"
            assert report.protocol_fingerprint != "none"
            assert report.environment["python"]["version"]
        finally:
            csv_path.unlink()


# ============================================================================
# Pipeline DAG Tests
# ============================================================================

class TestPipelineDAG:
    """Test pipeline DAG functionality"""

    def test_add_node(self):
        """Adding nodes should work"""
        dag = PipelineDAG()

        node = dag.add_node(
            "preprocess",
            NodeType.PREPROCESS,
            inputs=[],
            outputs=["X_processed"],
        )

        assert node.name == "preprocess"
        assert "preprocess" in dag.nodes

    def test_topological_sort(self):
        """Topological sort should order by dependencies"""
        dag = PipelineDAG()

        dag.add_node("preprocess", NodeType.PREPROCESS, inputs=[], outputs=["X"])
        dag.add_node("qc", NodeType.QC, inputs=["preprocess"], outputs=["qc_ok"])
        dag.add_node("features", NodeType.FEATURES, inputs=["qc"], outputs=["F"])
        dag.add_node("model", NodeType.MODEL, inputs=["features"], outputs=["metrics"])

        order = dag.topological_sort()

        assert order.index("preprocess") < order.index("qc")
        assert order.index("qc") < order.index("features")
        assert order.index("features") < order.index("model")

    def test_cycle_detection(self):
        """Cycles should be detected"""
        dag = PipelineDAG()

        dag.add_node("a", NodeType.PREPROCESS, inputs=["b"], outputs=[])
        dag.add_node("b", NodeType.QC, inputs=["a"], outputs=[])

        with pytest.raises(ValueError, match="Cycle"):
            dag.topological_sort()

    def test_missing_dependency(self):
        """Missing dependencies should be caught"""
        dag = PipelineDAG()

        dag.add_node("a", NodeType.PREPROCESS, inputs=["nonexistent"], outputs=[])

        with pytest.raises(ValueError, match="Dependency"):
            dag.topological_sort()

    def test_standard_pipeline(self):
        """Standard pipeline should be valid"""
        dag = build_standard_pipeline()

        assert len(dag.nodes) == 7  # 7 standard stages

        # Should be able to sort without error
        order = dag.get_execution_order()
        assert len(order) == 7


# ============================================================================
# Artifact Registry Tests
# ============================================================================

class TestArtifactRegistry:
    """Test artifact registry"""

    def test_register_artifact(self):
        """Registering artifacts should work"""
        registry = ArtifactRegistry()

        with tempfile.NamedTemporaryFile() as f:
            path = Path(f.name)

            artifact = registry.register(
                "test_metrics",
                ArtifactType.METRICS,
                path,
                description="Test metrics",
                source_node="model",
            )

            assert artifact.name == "test_metrics"
            assert artifact.artifact_type == ArtifactType.METRICS
            assert artifact.source_node == "model"

    def test_resolve_artifact(self):
        """Resolving artifacts should work"""
        registry = ArtifactRegistry()

        with tempfile.NamedTemporaryFile() as f:
            path = Path(f.name)
            registry.register("test", ArtifactType.PLOTS, path)

            artifact = registry.resolve("test")
            assert artifact is not None
            assert artifact.name == "test"

    def test_list_by_type(self):
        """Listing artifacts by type should work"""
        registry = ArtifactRegistry()

        with tempfile.NamedTemporaryFile() as f1, tempfile.NamedTemporaryFile() as f2:
            registry.register("m1", ArtifactType.METRICS, Path(f1.name))
            registry.register("m2", ArtifactType.METRICS, Path(f2.name))
            registry.register("p1", ArtifactType.PLOTS, Path(f1.name))

            by_type = registry.list_by_type()

            assert len(by_type["metrics"]) == 2
            assert len(by_type["plots"]) == 1

    def test_summary_stats(self):
        """Summary statistics should be accurate"""
        registry = ArtifactRegistry()

        with tempfile.NamedTemporaryFile() as f:
            registry.register("test", ArtifactType.METRICS, Path(f.name))

            summary = registry.summary()

            assert summary["total_artifacts"] == 1
            assert summary["count_by_type"]["metrics"] == 1


# ============================================================================
# Run Manifest Tests
# ============================================================================

class TestRunManifest:
    """Test run manifest generation"""

    def test_manifest_builder(self):
        """ManifestBuilder should construct manifests"""
        builder = ManifestBuilder("run_001")

        builder.set_protocol(
            ProtocolSnapshot(protocol_hash="abc123", task="authentication")
        )
        builder.set_data(
            DataSnapshot(data_fingerprint="def456", row_count=100)
        )
        builder.set_environment(
            EnvironmentSnapshot(
                seed=42,
                python_version="3.9",
                os_name="Linux",
                os_version="5.10",
                machine="x86_64",
                cpu_count=4,
            )
        )

        manifest = builder.build()

        assert manifest.metadata.run_id == "run_001"
        assert manifest.protocol.protocol_hash == "abc123"
        assert manifest.data.data_fingerprint == "def456"

    def test_manifest_to_json(self):
        """Manifest should serialize to JSON"""
        builder = ManifestBuilder("run_001")
        builder.set_protocol(ProtocolSnapshot(protocol_hash="abc"))
        builder.set_data(DataSnapshot(data_fingerprint="def"))
        builder.set_environment(
            EnvironmentSnapshot(
                seed=42,
                python_version="3.9",
                os_name="Linux",
                os_version="5.10",
                machine="x86_64",
                cpu_count=4,
            )
        )

        manifest = builder.build()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "manifest.json"
            manifest.to_json(json_path)

            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert data["metadata"]["run_id"] == "run_001"
            assert data["protocol"]["protocol_hash"] == "abc"


# ============================================================================
# Orchestrator Tests
# ============================================================================

class TestExecutionEngine:
    """Test central execution engine"""

    def test_engine_initialization(self):
        """Engine should initialize"""
        engine = ExecutionEngine()

        assert engine.run_id
        assert engine.manifest_builder
        assert engine.artifact_registry

    def test_global_seed_setting(self):
        """Engine should set global seed"""
        engine = ExecutionEngine()
        engine.setup_reproducibility(seed=42)

        assert get_global_seed() == 42

    def test_pipeline_setup(self):
        """Engine should setup pipeline"""
        engine = ExecutionEngine()
        dag = engine.setup_pipeline()

        assert dag is not None
        assert len(dag.nodes) > 0
        assert dag.pipeline_dag == engine.pipeline_dag

    def test_artifact_registration(self):
        """Engine should register artifacts"""
        engine = ExecutionEngine()

        with tempfile.NamedTemporaryFile() as f:
            engine.register_artifact(
                "test_metrics",
                ArtifactType.METRICS,
                Path(f.name),
            )

            artifact = engine.artifact_registry.resolve("test_metrics")
            assert artifact is not None

    def test_manifest_generation(self):
        """Engine should generate manifest"""
        engine = ExecutionEngine()

        # Setup minimal data
        protocol = MagicMock()
        protocol.task = "authentication"
        protocol.path = None

        engine.validate_protocol(protocol, {"task": "authentication"})

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w") as f:
            f.write("x,y\n1,2\n")
            f.flush()
            engine.validate_data(Path(f.name))

        engine.setup_reproducibility(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = engine.generate_manifest(Path(tmpdir))

            assert manifest_path.exists()
            assert engine.manifest is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests"""

    def test_philosophy_validation_complete(self):
        """Complete philosophy validation should work"""
        config = {"task": "authentication"}
        protocol = MagicMock()
        protocol.task = "authentication"
        protocol.modality = "Raman"
        protocol.model = "SVM"
        protocol.validation = "CV"

        qc_results = {"status": "pass", "pass_rate": 0.95}
        trust_outputs = {"calibration": {}, "conformal": {}}
        manifest = {
            "seed": 42,
            "python_version": "3.9",
            "os_info": "Linux",
            "package_versions": {},
            "data_fingerprint": "abc",
        }
        artifacts = {"report_html": Path("/tmp/report.html")}

        # Should not raise
        validate_all_principles(
            config,
            protocol,
            qc_results,
            trust_outputs,
            manifest,
            artifacts,
        )
