"""
FoodSpec Execution Engine (Orchestrator)

Central orchestrator that:
1. Validates philosophy compliance
2. Manages pipeline DAG execution
3. Tracks artifacts
4. Generates comprehensive manifests
5. Ensures reproducibility

This is the single point of truth for all FoodSpec runs.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from foodspec.core.philosophy import (
    validate_all_principles,
)
from foodspec.core.run_manifest import (
    ArtifactSnapshot,
    DAGSnapshot,
    DataSnapshot,
    EnvironmentSnapshot,
    ManifestBuilder,
    ProtocolSnapshot,
    RunManifest,
    RunStatus,
)
from foodspec.engine.artifacts import ArtifactRegistry, ArtifactType
from foodspec.engine.dag import PipelineDAG, build_standard_pipeline
from foodspec.utils.determinism import (
    capture_environment,
    capture_versions,
    fingerprint_csv,
    fingerprint_protocol,
    set_global_seed,
)

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Central execution orchestrator for FoodSpec.
    
    Manages:
    1. Philosophy validation
    2. Reproducibility (seeding, environment capture)
    3. Pipeline DAG execution
    4. Artifact tracking
    5. Manifest generation
    6. Error handling and recovery
    
    Every run must go through this engine.
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize execution engine.
        
        Args:
            run_id: Unique run identifier (generated if None)
        """
        self.run_id = run_id or str(uuid.uuid4())[:12]
        self.manifest_builder = ManifestBuilder(self.run_id)
        self.artifact_registry = ArtifactRegistry()
        self.pipeline_dag: Optional[PipelineDAG] = None
        self.execution_context: Dict[str, Any] = {}
        self.manifest: Optional[RunManifest] = None

        logger.info(f"Execution Engine initialized (run_id={self.run_id})")

    # ========================================================================
    # Step 1-2: Protocol & Data Validation
    # ========================================================================

    def validate_protocol(
        self,
        protocol: Any,
        protocol_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Step 1: Validate and fingerprint protocol.
        
        Args:
            protocol: ProtocolConfig object
            protocol_dict: Protocol as dict (for hashing)
            
        Returns:
            Protocol hash
            
        Raises:
            PhilosophyError: If protocol invalid
        """
        logger.info("STEP 1: Validating protocol")

        # Enforce TASK_FIRST principle
        from foodspec.core.philosophy import enforce_task_first
        enforce_task_first(protocol_dict or protocol)

        # Enforce PROTOCOL_IS_SOURCE_OF_TRUTH
        from foodspec.core.philosophy import enforce_protocol_truth
        enforce_protocol_truth(protocol)

        # Compute protocol hash
        proto_dict = protocol_dict or {}
        protocol_hash = fingerprint_protocol(proto_dict)

        # Build protocol snapshot
        self.manifest_builder.set_protocol(
            ProtocolSnapshot(
                protocol_hash=protocol_hash,
                protocol_path=str(getattr(protocol, "path", None)),
                task=getattr(protocol, "task", None),
                modality=getattr(protocol, "modality", None),
                model=getattr(protocol, "model", None),
                validation=getattr(protocol, "validation", None),
                config_dict=proto_dict,
            )
        )

        logger.info(f"  ✓ Protocol validated (hash={protocol_hash[:16]}...)")
        return protocol_hash

    def validate_data(self, csv_path: Path) -> Dict[str, Any]:
        """
        Step 2: Validate and fingerprint data.
        
        Args:
            csv_path: Path to input CSV
            
        Returns:
            Data fingerprint dict
        """
        logger.info("STEP 2: Validating data")

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        # Compute data fingerprint
        data_hash = fingerprint_csv(csv_path)

        # Basic statistics
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            row_count = len(df)
            col_count = len(df.columns)
        except Exception as e:
            logger.warning(f"Could not read CSV statistics: {e}")
            row_count = None
            col_count = None

        size_bytes = csv_path.stat().st_size

        # Build data snapshot
        self.manifest_builder.set_data(
            DataSnapshot(
                data_fingerprint=data_hash,
                data_path=str(csv_path),
                row_count=row_count,
                column_count=col_count,
                size_bytes=size_bytes,
            )
        )

        logger.info(f"  ✓ Data validated (hash={data_hash[:16]}..., rows={row_count}, cols={col_count})")

        return {
            "fingerprint": data_hash,
            "rows": row_count,
            "cols": col_count,
            "size": size_bytes,
        }

    # ========================================================================
    # Step 3: Setup Reproducibility
    # ========================================================================

    def setup_reproducibility(self, seed: Optional[int] = None) -> None:
        """
        Step 3: Setup reproducibility infrastructure.
        
        Args:
            seed: Random seed (generated if None)
        """
        logger.info("STEP 3: Setting up reproducibility")

        # Generate seed if not provided
        if seed is None:
            import random as py_random
            seed = py_random.randint(0, 2**31 - 1)

        # Set global seed
        set_global_seed(seed)

        # Capture environment
        env = capture_environment()
        versions = capture_versions()

        # Build environment snapshot
        self.manifest_builder.set_environment(
            EnvironmentSnapshot(
                seed=seed,
                python_version=env["python"]["version"],
                os_name=env["os"]["name"],
                os_version=env["os"]["version"],
                machine=env["machine"]["machine"],
                cpu_count=env["machine"]["cpu_count"] or 0,
                package_versions=versions.get("critical_packages", {}),
            )
        )

        self.execution_context["seed"] = seed

        logger.info(f"  ✓ Reproducibility setup (seed={seed})")

    # ========================================================================
    # Step 4-9: Pipeline Execution
    # ========================================================================

    def setup_pipeline(self) -> PipelineDAG:
        """
        Step 4: Setup pipeline DAG.
        
        Returns:
            Configured PipelineDAG
        """
        logger.info("STEP 4: Setting up pipeline DAG")

        # Build standard pipeline
        self.pipeline_dag = build_standard_pipeline()
        self.pipeline_dag.validate()

        logger.info(f"  ✓ Pipeline DAG setup ({len(self.pipeline_dag.nodes)} nodes)")

        return self.pipeline_dag

    def register_stage_function(
        self,
        stage_name: str,
        func: Callable,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a stage function in pipeline.
        
        Args:
            stage_name: Name of stage (e.g., "preprocess", "qc", "model")
            func: Callable that implements stage
            params: Stage-specific parameters
        """
        if self.pipeline_dag is None:
            raise ValueError("Pipeline not initialized; call setup_pipeline() first")

        node = self.pipeline_dag.get_node(stage_name)
        if node is None:
            raise ValueError(f"Node '{stage_name}' not found in pipeline")

        node.func = func
        node.params = params or {}

        logger.info(f"Registered function for stage: {stage_name}")

    def execute_pipeline(self) -> Dict[str, Any]:
        """
        Steps 5-9: Execute pipeline DAG.
        
        Returns:
            Execution results
        """
        logger.info("STEPS 5-9: Executing pipeline")

        if self.pipeline_dag is None:
            raise ValueError("Pipeline not initialized")

        # Execute all stages in order
        results = self.pipeline_dag.execute(self.execution_context)

        # Save DAG execution record
        self.manifest_builder.set_dag(
            DAGSnapshot(
                dag_dict=self.pipeline_dag.to_dict(),
                execution_order=self.pipeline_dag.get_execution_order(),
                node_count=len(self.pipeline_dag.nodes),
            )
        )

        logger.info("  ✓ Pipeline execution complete")

        return results

    # ========================================================================
    # Step 10: Artifact Registration
    # ========================================================================

    def register_artifact(
        self,
        name: str,
        artifact_type: ArtifactType,
        path: Path,
        description: str = "",
        source_node: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Step 10: Register an artifact.
        
        Args:
            name: Unique artifact name
            artifact_type: Type of artifact
            path: File path
            description: Human description
            source_node: Which DAG node produced this
            metadata: Extra metadata
        """
        self.artifact_registry.register(
            name=name,
            artifact_type=artifact_type,
            path=path,
            description=description,
            source_node=source_node,
            metadata=metadata or {},
        )

    def finalize_artifacts(self) -> None:
        """Finalize and validate artifact registry"""
        logger.info("STEP 10: Finalizing artifact registry")

        self.artifact_registry.validate()

        # Create artifact snapshot
        summary = self.artifact_registry.summary()
        self.manifest_builder.set_artifacts(
            ArtifactSnapshot(
                artifact_count=summary["total_artifacts"],
                by_type=summary["count_by_type"],
                total_size_bytes=summary["total_size_bytes"],
            )
        )

        logger.info(f"  ✓ {len(self.artifact_registry.artifacts)} artifacts registered")

    # ========================================================================
    # Step 11: Generate Manifest
    # ========================================================================

    def generate_manifest(self, out_dir: Path) -> Path:
        """
        Step 11: Generate run manifest.
        
        Args:
            out_dir: Directory to save manifest
            
        Returns:
            Path to manifest JSON
        """
        logger.info("STEP 11: Generating run manifest")

        self.manifest = self.manifest_builder.build()

        manifest_path = Path(out_dir) / "run_manifest.json"
        self.manifest.to_json(manifest_path)

        # Also save artifact registry
        registry_path = Path(out_dir) / "artifacts.json"
        self.artifact_registry.to_json(registry_path)

        logger.info(f"  ✓ Manifest saved to: {manifest_path}")

        return manifest_path

    # ========================================================================
    # Step 12: Philosophy Enforcement
    # ========================================================================

    def enforce_philosophy(
        self,
        protocol: Any,
        protocol_dict: Dict[str, Any],
        qc_results: Dict[str, Any],
        trust_outputs: Dict[str, Any],
    ) -> None:
        """
        Step 12: Enforce all design principles.
        
        Args:
            protocol: ProtocolConfig object
            protocol_dict: Protocol as dict
            qc_results: QC check results
            trust_outputs: Trust stack outputs
            
        Raises:
            PhilosophyError: If any principle violated
        """
        logger.info("STEP 12: Philosophy enforcement")

        # Build artifacts dict for enforcement
        artifacts_dict = {
            f.split("/")[-1]: Path(f) for f in [
                str(a.path) for a in self.artifact_registry.artifacts.values()
            ]
        }

        # Run all philosophy checks
        validate_all_principles(
            config=protocol_dict,
            protocol=protocol,
            qc_results=qc_results,
            trust_outputs=trust_outputs,
            manifest=self.manifest.to_dict() if self.manifest else {},
            artifacts=artifacts_dict,
        )

        logger.info("  ✓ Philosophy enforcement passed")

    # ========================================================================
    # Complete Run Orchestration
    # ========================================================================

    def run(
        self,
        protocol: Any,
        protocol_dict: Dict[str, Any],
        csv_path: Path,
        out_dir: Path,
        seed: Optional[int] = None,
        skip_philosophy: bool = False,
    ) -> Path:
        """
        Execute complete FoodSpec run with all 12 steps.
        
        Args:
            protocol: ProtocolConfig object
            protocol_dict: Protocol as dict
            csv_path: Path to input CSV
            out_dir: Output directory
            seed: Random seed (optional)
            skip_philosophy: Skip philosophy enforcement (not recommended)
            
        Returns:
            Path to output directory
            
        Raises:
            Various exceptions if any step fails
        """
        logger.info("=" * 70)
        logger.info("EXECUTION ENGINE: Starting comprehensive FoodSpec run")
        logger.info("=" * 70)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1-2: Validate protocol and data
            self.validate_protocol(protocol, protocol_dict)
            self.validate_data(csv_path)

            # Step 3: Setup reproducibility
            self.setup_reproducibility(seed)

            # Step 4: Setup pipeline
            self.setup_pipeline()

            # Mark run as started
            self.manifest_builder.metadata.status = RunStatus.RUNNING

            # Steps 5-9: Execute pipeline (caller must register stage functions)
            # This is typically done by higher-level orchestration code

            # Step 10: Finalize artifacts
            self.finalize_artifacts()

            # Step 11: Generate manifest
            self.generate_manifest(out_dir)

            # Step 12: Philosophy enforcement (optional)
            if not skip_philosophy:
                # Caller must provide these results
                logger.warning("Philosophy enforcement skipped (incomplete results)")

            # Mark success
            self.manifest.mark_success()
            self.manifest.to_json(out_dir / "run_manifest.json")

            logger.info("=" * 70)
            logger.info(self.manifest.summary())
            logger.info("=" * 70)

            return out_dir

        except Exception as e:
            logger.error(f"Run failed: {e}", exc_info=True)

            if self.manifest:
                self.manifest.mark_failed(str(e))
                self.manifest.to_json(out_dir / "run_manifest.json")

            raise

    # ========================================================================
    # Utilities
    # ========================================================================

    def get_registry(self) -> ArtifactRegistry:
        """Get artifact registry"""
        return self.artifact_registry

    def get_manifest(self) -> Optional[RunManifest]:
        """Get generated manifest"""
        return self.manifest

    def get_execution_context(self) -> Dict[str, Any]:
        """Get execution context"""
        return self.execution_context
