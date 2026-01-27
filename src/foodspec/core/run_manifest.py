"""
FoodSpec Run Manifest

Comprehensive metadata for every run.
Source of truth for reproducibility and provenance.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def _json_safe(value: Any) -> Any:
    """Coerce values into JSON-serializable forms.

    MagicMocks or other non-serializable objects are stringified to keep
    manifest generation resilient during testing and debugging.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """Status of a run"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ManifestMetadata:
    """Top-level run metadata"""

    run_id: str                                     # Unique run identifier
    timestamp_start: str                            # ISO format
    timestamp_end: Optional[str] = None             # ISO format
    status: RunStatus = RunStatus.PENDING
    version: str = "1.0"                           # Manifest version
    foodspec_version: str = "unknown"               # FoodSpec version

    def duration_seconds(self) -> Optional[float]:
        """Compute run duration"""
        if not self.timestamp_end:
            return None
        from datetime import datetime
        start = datetime.fromisoformat(self.timestamp_start)
        end = datetime.fromisoformat(self.timestamp_end)
        return (end - start).total_seconds()


@dataclass
class ProtocolSnapshot:
    """Snapshot of protocol configuration"""

    protocol_hash: str                              # SHA256 of protocol
    protocol_path: Optional[str] = None             # Where protocol came from
    task: Optional[str] = None
    modality: Optional[str] = None
    model: Optional[str] = None
    validation: Optional[str] = None
    config_dict: Optional[Dict[str, Any]] = None    # Full protocol dict


@dataclass
class DataSnapshot:
    """Snapshot of input data"""

    data_fingerprint: str                           # SHA256 of CSV
    data_path: Optional[str] = None                 # Where data came from
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    size_bytes: Optional[int] = None


@dataclass
class EnvironmentSnapshot:
    """Capture of execution environment"""

    seed: Optional[int]                             # Random seed
    python_version: str                             # Python version
    os_name: str                                    # OS type
    os_version: str                                 # OS version
    machine: str                                    # Machine type
    cpu_count: int                                  # CPU cores
    package_versions: Dict[str, str] = field(default_factory=dict)  # Key packages


@dataclass
class DAGSnapshot:
    """Snapshot of pipeline DAG"""

    dag_dict: Dict[str, Any]                        # Full DAG JSON
    execution_order: List[str]                      # Node execution order
    node_count: int                                 # Number of nodes


@dataclass
class ArtifactSnapshot:
    """Snapshot of produced artifacts"""

    artifact_count: int                             # Total artifacts
    by_type: Dict[str, int]                         # Count by type
    total_size_bytes: int                           # Total size


@dataclass
class RunManifest:
    """
    Comprehensive run manifest.
    
    Contains all metadata needed for reproducibility and provenance.
    """

    metadata: ManifestMetadata
    protocol: ProtocolSnapshot
    data: DataSnapshot
    environment: EnvironmentSnapshot
    dag: Optional[DAGSnapshot] = None
    artifacts: Optional[ArtifactSnapshot] = None
    philosophy_checks: Optional[Dict[str, bool]] = None  # Philosophy validation results
    errors: List[str] = field(default_factory=list)      # Any errors during run

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (preserves structure)"""
        return {
            "metadata": asdict(self.metadata),
            "protocol": asdict(self.protocol),
            "data": asdict(self.data),
            "environment": asdict(self.environment),
            "dag": asdict(self.dag) if self.dag else None,
            "artifacts": asdict(self.artifacts) if self.artifacts else None,
            "philosophy_checks": self.philosophy_checks,
            "errors": self.errors,
        }

    def to_json(self, out_path: Path) -> Path:
        """Save manifest to JSON file"""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects
        manifest_dict = _json_safe(self.to_dict())

        with open(out_path, "w") as f:
            json.dump(manifest_dict, f, indent=2)

        logger.info(f"Manifest saved to: {out_path}")
        return out_path

    def mark_running(self) -> None:
        """Mark run as started"""
        self.metadata.status = RunStatus.RUNNING
        logger.info("Run marked as RUNNING")

    def mark_success(self) -> None:
        """Mark run as completed successfully"""
        self.metadata.status = RunStatus.SUCCESS
        self.metadata.timestamp_end = datetime.utcnow().isoformat()
        duration = self.metadata.duration_seconds()
        logger.info(f"Run marked as SUCCESS (duration: {duration:.1f}s)")

    def mark_failed(self, error: str) -> None:
        """Mark run as failed"""
        self.metadata.status = RunStatus.FAILED
        self.metadata.timestamp_end = datetime.utcnow().isoformat()
        self.errors.append(error)
        logger.error(f"Run marked as FAILED: {error}")

    def add_error(self, error: str) -> None:
        """Record an error (doesn't change status)"""
        self.errors.append(error)
        logger.warning(f"Error recorded: {error}")

    def summary(self) -> str:
        """Generate human-readable summary"""
        duration = self.metadata.duration_seconds()
        duration_str = f"{duration:.1f}s" if duration else "in progress"

        summary = f"""
╭─ RUN MANIFEST SUMMARY ──────────────────────────────────╮
│ Run ID:              {self.metadata.run_id}
│ Status:              {self.metadata.status.value}
│ Duration:            {duration_str}
│ FoodSpec Version:    {self.metadata.foodspec_version}
│
│ Protocol:
│   - Hash:            {self.protocol.protocol_hash[:16]}...
│   - Task:            {self.protocol.task}
│   - Model:           {self.protocol.model}
│
│ Data:
│   - Fingerprint:     {self.data.data_fingerprint[:16]}...
│   - Rows:            {self.data.row_count}
│   - Cols:            {self.data.column_count}
│   - Size:            {self.data.size_bytes / 1024:.1f} KB
│
│ Environment:
│   - Python:          {self.environment.python_version}
│   - Seed:            {self.environment.seed}
│   - OS:              {self.environment.os_name} {self.environment.os_version}
│   - CPUs:            {self.environment.cpu_count}
│
│ Artifacts:
│   - Count:           {self.artifacts.artifact_count if self.artifacts else 0}
│   - Size:            {self.artifacts.total_size_bytes / 1024 / 1024:.1f} MB if self.artifacts else 0
│
│ Errors:              {len(self.errors)}
╰──────────────────────────────────────────────────────────╯
"""
        return summary


# ============================================================================
# Manifest Builder
# ============================================================================

class ManifestBuilder:
    """Fluent builder for constructing manifests"""

    def __init__(self, run_id: str):
        self.metadata = ManifestMetadata(
            run_id=run_id,
            timestamp_start=datetime.utcnow().isoformat(),
        )
        self.protocol: Optional[ProtocolSnapshot] = None
        self.data: Optional[DataSnapshot] = None
        self.environment: Optional[EnvironmentSnapshot] = None
        self.dag: Optional[DAGSnapshot] = None
        self.artifacts: Optional[ArtifactSnapshot] = None
        self.philosophy_checks: Dict[str, bool] = {}
        self.errors: List[str] = []

    def set_foodspec_version(self, version: str) -> "ManifestBuilder":
        """Set FoodSpec version"""
        self.metadata.foodspec_version = version
        return self

    def set_protocol(self, protocol_snapshot: ProtocolSnapshot) -> "ManifestBuilder":
        """Set protocol snapshot"""
        self.protocol = protocol_snapshot
        return self

    def set_data(self, data_snapshot: DataSnapshot) -> "ManifestBuilder":
        """Set data snapshot"""
        self.data = data_snapshot
        return self

    def set_environment(self, env_snapshot: EnvironmentSnapshot) -> "ManifestBuilder":
        """Set environment snapshot"""
        self.environment = env_snapshot
        return self

    def set_dag(self, dag_snapshot: DAGSnapshot) -> "ManifestBuilder":
        """Set DAG snapshot"""
        self.dag = dag_snapshot
        return self

    def set_artifacts(self, artifact_snapshot: ArtifactSnapshot) -> "ManifestBuilder":
        """Set artifact snapshot"""
        self.artifacts = artifact_snapshot
        return self

    def record_philosophy_check(self, check_name: str, passed: bool) -> "ManifestBuilder":
        """Record philosophy check result"""
        self.philosophy_checks[check_name] = passed
        return self

    def add_error(self, error: str) -> "ManifestBuilder":
        """Add error"""
        self.errors.append(error)
        return self

    def build(self) -> RunManifest:
        """Build and return manifest"""
        if self.protocol is None:
            raise ValueError("Protocol snapshot required")
        if self.data is None:
            raise ValueError("Data snapshot required")
        if self.environment is None:
            raise ValueError("Environment snapshot required")

        return RunManifest(
            metadata=self.metadata,
            protocol=self.protocol,
            data=self.data,
            environment=self.environment,
            dag=self.dag,
            artifacts=self.artifacts,
            philosophy_checks=self.philosophy_checks if self.philosophy_checks else None,
            errors=self.errors,
        )
