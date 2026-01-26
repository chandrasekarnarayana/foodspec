"""
FoodSpec Artifact Registry

Centralized tracking of all run outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from enum import Enum
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ArtifactType(str, Enum):
    """Types of artifacts produced during run"""
    METRICS = "metrics"
    PREDICTIONS = "predictions"
    PLOTS = "plots"
    MODELS = "models"
    REPORTS = "reports"
    QC = "qc"
    TRUST = "trust"
    INTERMEDIATE = "intermediate"
    MANIFEST = "manifest"
    PROVENANCE = "provenance"


@dataclass
class Artifact:
    """
    Single artifact in registry.
    
    Represents one output file/object.
    """
    
    name: str                               # Unique name
    artifact_type: ArtifactType             # Type
    path: Path                              # File path
    created_at: str = None                  # ISO timestamp
    size_bytes: Optional[int] = None        # File size
    description: str = ""                   # Human description
    source_node: Optional[str] = None       # Which DAG node produced this
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra metadata
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.size_bytes is None and self.path and Path(self.path).exists():
            self.size_bytes = Path(self.path).stat().st_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dict"""
        return {
            "name": self.name,
            "type": self.artifact_type.value,
            "path": str(self.path),
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "description": self.description,
            "source_node": self.source_node,
            "metadata": self.metadata,
        }


@dataclass
class ArtifactRegistry:
    """
    Central registry for all run artifacts.
    
    Tracks:
    - All output files
    - File types and purposes
    - Provenance (which stage produced each)
    - Metadata
    """
    
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    types: Dict[ArtifactType, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize type indices
        for art_type in ArtifactType:
            self.types[art_type] = []
    
    def register(
        self,
        name: str,
        artifact_type: ArtifactType,
        path: Path,
        description: str = "",
        source_node: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """
        Register an artifact.
        
        Args:
            name: Unique name
            artifact_type: Type of artifact
            path: File path
            description: Human description
            source_node: Which DAG node produced this
            metadata: Extra metadata
            
        Returns:
            Registered Artifact
        """
        if name in self.artifacts:
            logger.warning(f"Artifact '{name}' already registered, overwriting")
        
        artifact = Artifact(
            name=name,
            artifact_type=artifact_type,
            path=Path(path),
            description=description,
            source_node=source_node,
            metadata=metadata or {},
        )
        
        self.artifacts[name] = artifact
        
        # Update type index
        if name not in self.types[artifact_type]:
            self.types[artifact_type].append(name)
        
        logger.info(f"Registered artifact: {name} ({artifact_type.value}) → {path}")
        
        return artifact
    
    def resolve(self, name: str) -> Optional[Artifact]:
        """Get artifact by name"""
        return self.artifacts.get(name)
    
    def resolve_by_type(self, artifact_type: ArtifactType) -> List[Artifact]:
        """Get all artifacts of a type"""
        names = self.types.get(artifact_type, [])
        return [self.artifacts[n] for n in names if n in self.artifacts]
    
    def list_all(self) -> List[Artifact]:
        """List all artifacts"""
        return list(self.artifacts.values())
    
    def list_by_type(self) -> Dict[str, List[str]]:
        """List artifact names grouped by type"""
        return {
            art_type.value: names
            for art_type, names in self.types.items()
            if names
        }
    
    def list_by_source(self) -> Dict[str, List[str]]:
        """List artifact names grouped by source node"""
        by_source: Dict[str, List[str]] = {}
        for name, artifact in self.artifacts.items():
            source = artifact.source_node or "unknown"
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(name)
        return by_source
    
    def count_by_type(self) -> Dict[str, int]:
        """Count artifacts by type"""
        return {
            art_type.value: len(names)
            for art_type, names in self.types.items()
        }
    
    def total_size(self) -> int:
        """Compute total size of all artifacts in bytes"""
        return sum(a.size_bytes or 0 for a in self.artifacts.values())
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            "total_artifacts": len(self.artifacts),
            "count_by_type": self.count_by_type(),
            "total_size_bytes": self.total_size(),
            "artifacts_by_type": self.list_by_type(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry to dict"""
        return {
            "artifacts": {name: art.to_dict() for name, art in self.artifacts.items()},
            "summary": self.summary(),
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def to_json(self, out_path: Path) -> Path:
        """Save registry to JSON file"""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Registry saved to: {out_path}")
        logger.info(f"  → {len(self.artifacts)} artifacts tracked")
        logger.info(f"  → {len(self.count_by_type())} types")
        logger.info(f"  → {self.total_size() / 1024 / 1024:.1f} MB total")
        
        return out_path
    
    def validate(self) -> bool:
        """
        Validate registry consistency.
        
        Checks:
        - All artifact paths exist
        - No duplicate names
        - Size metadata valid
        
        Returns:
            True if valid
        """
        issues = []
        
        for name, artifact in self.artifacts.items():
            # Check path exists
            if not artifact.path.exists():
                issues.append(f"Artifact '{name}' path does not exist: {artifact.path}")
            
            # Check size is positive or None
            if artifact.size_bytes is not None and artifact.size_bytes < 0:
                issues.append(f"Artifact '{name}' has negative size: {artifact.size_bytes}")
        
        if issues:
            for issue in issues:
                logger.error(f"  ✗ {issue}")
            raise ValueError(f"Registry validation failed: {len(issues)} issues")
        
        logger.info("✓ Registry is valid")
        return True
    
    def export_manifest(self, out_path: Path) -> Path:
        """
        Export artifact manifest for provenance tracking.
        
        Manifest lists all artifacts with their provenance.
        """
        manifest = {
            "artifacts": [a.to_dict() for a in self.artifacts.values()],
            "summary": self.summary(),
            "by_source": self.list_by_source(),
            "generated_at": datetime.utcnow().isoformat(),
        }
        
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Artifact manifest saved to: {out_path}")
        return out_path


# ============================================================================
# Registry Access Functions
# ============================================================================

_GLOBAL_REGISTRY: Optional[ArtifactRegistry] = None


def get_registry() -> ArtifactRegistry:
    """Get global artifact registry"""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ArtifactRegistry()
    return _GLOBAL_REGISTRY


def set_registry(registry: ArtifactRegistry) -> None:
    """Set global artifact registry"""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = registry


def reset_registry() -> None:
    """Reset global registry (useful for testing)"""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None


def register_artifact(
    name: str,
    artifact_type: ArtifactType,
    path: Path,
    **kwargs,
) -> Artifact:
    """Register artifact in global registry"""
    registry = get_registry()
    return registry.register(name, artifact_type, path, **kwargs)


def resolve_artifact(name: str) -> Optional[Artifact]:
    """Resolve artifact from global registry"""
    registry = get_registry()
    return registry.resolve(name)
