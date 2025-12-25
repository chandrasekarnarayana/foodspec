"""
Feature & model registry for provenance and audit trails.
Stores entries in a JSON index (could be swapped for SQLite later).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hash_dataset(df) -> str:
    """Compute SHA-256 hash of dataset for provenance tracking.
    
    Generates a deterministic fingerprint of the dataset to detect data drift,
    enable reproducibility audits, and verify model-data alignment.
    
    **Reproducibility Context:**
    Dataset hash is critical for FAIR principles (Findable, Accessible,
    Interoperable, Reusable). Ensures models trained on specific data versions
    can be traced back to exact input, detecting silent data corruption or
    preprocessing changes.
    
    **Significance:**
    - Hash mismatch indicates data changed → model predictions unreliable
    - Use for version control of datasets (like Git for data)
    - Required for regulatory compliance (21 CFR Part 11, ISO 17025)
    
    Args:
        df: pandas DataFrame with spectral data (rows=samples, cols=features)
    
    Returns:
        SHA-256 hash (64 hex characters) or "unknown" if hashing fails
    
    Examples:
        >>> import pandas as pd
        >>> from foodspec.registry import _hash_dataset
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> hash1 = _hash_dataset(df)
        >>> hash2 = _hash_dataset(df)  # Same data → same hash
        >>> assert hash1 == hash2
        >>> df.loc[0, 'A'] = 999  # Change data
        >>> assert _hash_dataset(df) != hash1  # Hash detects change
    
    See Also:
        - Theory: docs/foundations/data_structures_and_fair_principles.md
        - Reproducibility: docs/protocols/reference_protocol.md#provenance
    """
    try:
        return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
    except Exception:
        return "unknown"


@dataclass
class RegistryEntry:
    """Immutable record of a single analysis run or trained model.
    
    Stores complete provenance for reproducibility: dataset fingerprint,
    preprocessing steps, feature extraction, model details, and validation
    metrics. Enables audit trails, model comparison, and error tracking.
    
    **Reproducibility Requirements:**
    - `dataset_hash`: Detect data drift (hash mismatch = data changed)
    - `protocol_version`: Semantic versioning for method reproducibility
    - `provenance`: Timestamp, user, tool version for audit trails
    - `validation_strategy`: CV method, test set size for fair comparison
    
    **Compliance Use Cases:**
    - FDA 21 CFR Part 11: Electronic records and signatures
    - ISO 17025: Testing and calibration laboratory competence
    - GLP (Good Laboratory Practice): Data integrity and traceability
    
    Attributes:
        dataset_hash: SHA-256 of input data (64 hex chars)
        protocol_name: Canonical protocol identifier (e.g., "oil_authentication_v2")
        protocol_version: Semantic version (e.g., "2.1.0")
        preprocessing: Dict of preprocessing steps and parameters
        features: List of feature extraction configs
        model_id: Unique model identifier (optional)
        metrics: Performance metrics dict (e.g., {'auc': 0.95, 'f1': 0.89})
        provenance: Metadata dict (timestamp, user, tool_version)
        run_id: Unique analysis run identifier (optional)
        validation_strategy: CV method (e.g., "5-fold", "stratified_kfold")
        model_path: File path to saved model (optional)
        model_type: Model class (e.g., "PLSRegression", "RandomForestClassifier")
        inputs: List of input file paths
    
    Examples:
        >>> from foodspec.registry import RegistryEntry
        >>> entry = RegistryEntry(
        ...     dataset_hash="abc123...",
        ...     protocol_name="oil_auth",
        ...     protocol_version="1.0.0",
        ...     preprocessing={"baseline": "als", "lambda": 100},
        ...     features=[{"name": "rq_1743_1654", "type": "ratio"}],
        ...     metrics={"auc": 0.95},
        ...     provenance={"timestamp": "2025-12-25T10:00:00Z"}
        ... )
        >>> assert entry.dataset_hash == "abc123..."
    
    See Also:
        - Theory: docs/foundations/data_structures_and_fair_principles.md
        - Protocols: docs/protocols/reference_protocol.md#step-8-document
    """
    dataset_hash: str
    protocol_name: str
    protocol_version: str
    preprocessing: Dict[str, Any]
    features: List[Dict[str, Any]]
    model_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    validation_strategy: Optional[str] = None
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    inputs: List[str] = field(default_factory=list)


class FeatureModelRegistry:
    """Persistent JSON-based registry for analysis runs and trained models.
    
    Maintains audit trail of all FoodSpec analyses for reproducibility,
    model comparison, and regulatory compliance. Tracks dataset versions,
    preprocessing pipelines, feature extraction, and validation metrics.
    
    **Design Philosophy:**
    - Single source of truth for all runs (no scattered Excel files)
    - Immutable append-only log (never modify past entries)
    - FAIR-compliant provenance (timestamp, user, tool version)
    - Lightweight JSON format (can migrate to SQLite if >10,000 entries)
    
    **When to Use:**
    - Track model performance over time (detect degradation)
    - Compare preprocessing methods on same dataset
    - Reproduce published results (match run_id from paper)
    - Audit trail for quality control labs
    
    Attributes:
        path: Path to JSON registry file (created if not exists)
        entries: List of RegistryEntry objects loaded from file
    
    Examples:
        >>> from pathlib import Path
        >>> from foodspec.registry import FeatureModelRegistry
        >>> registry = FeatureModelRegistry(Path("runs.json"))
        >>> print(f"Total runs: {len(registry.entries)}")
        >>> oils = registry.query_by_protocol("oil_authentication")
        >>> print(f"Oil runs: {len(oils)}")
    
    See Also:
        - Theory: docs/foundations/data_structures_and_fair_principles.md
        - Tutorial: docs/02-tutorials/reference_analysis_oil_authentication.md#step-8
    """
    
    def __init__(self, path: Path):
        """Initialize registry from JSON file or create empty registry.
        
        Args:
            path: Path to JSON registry file (e.g., Path("registry.json"))
        
        Raises:
            None: Silently creates empty registry if file corrupted/missing
        """
        self.path = Path(path)
        self.entries: List[RegistryEntry] = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self.entries = [RegistryEntry(**e) for e in raw]
            except Exception:
                self.entries = []

    def add_entry(self, entry: RegistryEntry):
        """Append new entry to registry and persist to disk.
        
        Immutable append-only operation (never modifies existing entries).
        Atomically writes to disk after each addition for durability.
        
        Args:
            entry: RegistryEntry to add (must be complete, no validation)
        
        Examples:
            >>> from foodspec.registry import RegistryEntry, FeatureModelRegistry
            >>> from pathlib import Path
            >>> registry = FeatureModelRegistry(Path("runs.json"))
            >>> entry = RegistryEntry(
            ...     dataset_hash="abc123",
            ...     protocol_name="test",
            ...     protocol_version="1.0.0",
            ...     preprocessing={},
            ...     features=[]
            ... )
            >>> registry.add_entry(entry)
            >>> assert len(registry.entries) > 0
        
        See Also:
            - register_run(): Higher-level API for analysis runs
            - register_model(): Higher-level API for trained models
        """
        self.entries.append(entry)
        self._save()

    def register_run(self, run_id: str, metadata: Dict[str, Any]):
        """
        Lightweight run registration (without model).
        """
        entry = RegistryEntry(
            dataset_hash=metadata.get("dataset_hash", "unknown"),
            protocol_name=metadata.get("protocol", "unknown"),
            protocol_version=metadata.get("protocol_version", "unknown"),
            preprocessing=metadata.get("preprocessing", {}),
            features=metadata.get("features", []),
            metrics=metadata.get("metrics", {}),
            provenance={
                "timestamp": metadata.get("timestamp"),
                "user": metadata.get("user"),
                "tool_version": metadata.get("tool_version"),
            },
            run_id=run_id,
            validation_strategy=metadata.get("validation_strategy"),
            inputs=metadata.get("inputs", []),
        )
        self.add_entry(entry)

    def register_model(self, run_id: str, model_path: str, model_metadata: Dict[str, Any]):
        """
        Log a model produced in a run.
        """
        entry = RegistryEntry(
            dataset_hash=model_metadata.get("dataset_hash", "unknown"),
            protocol_name=model_metadata.get("protocol_name", "unknown"),
            protocol_version=model_metadata.get("protocol_version", "unknown"),
            preprocessing=model_metadata.get("preprocessing", {}),
            features=model_metadata.get("features", []),
            model_id=model_metadata.get("model_id", Path(model_path).stem),
            model_path=model_path,
            model_type=model_metadata.get("model_type"),
            metrics=model_metadata.get("metrics", {}),
            provenance={
                "timestamp": model_metadata.get("timestamp"),
                "user": model_metadata.get("user"),
                "tool_version": model_metadata.get("tool_version"),
            },
            run_id=run_id,
            validation_strategy=model_metadata.get("validation_strategy"),
            inputs=model_metadata.get("inputs", []),
        )
        self.add_entry(entry)

    def _save(self):
        """Persist all entries to JSON file.
        
        Internal method called automatically after add_entry(). Serializes
        all RegistryEntry objects to human-readable JSON with 2-space indent.
        
        **Durability Note:**
        Not transactional (no rollback on error). Consider using SQLite
        if concurrent writes or >10,000 entries expected.
        
        Raises:
            IOError: If disk write fails (full disk, permissions)
        """
        self.path.write_text(json.dumps([asdict(e) for e in self.entries], indent=2), encoding="utf-8")

    def query_by_feature(self, feature_name: str) -> List[RegistryEntry]:
        """Find all runs using a specific feature.
        
        Useful for comparing preprocessing methods (e.g., "rq_1743_1654"
        performance across ALS vs. polynomial baseline correction).
        
        Args:
            feature_name: Exact feature identifier (e.g., "rq_1743_1654")
        
        Returns:
            List of RegistryEntry objects using this feature (empty if none)
        
        Examples:
            >>> from foodspec.registry import FeatureModelRegistry
            >>> from pathlib import Path
            >>> registry = FeatureModelRegistry(Path("runs.json"))
            >>> rq_runs = registry.query_by_feature("rq_1743_1654")
            >>> aucs = [r.metrics.get('auc', 0) for r in rq_runs]
            >>> print(f"RQ feature AUC range: {min(aucs):.2f}-{max(aucs):.2f}")
        
        See Also:
            - query_by_protocol(): Filter by protocol instead
        """
        return [e for e in self.entries if any(f.get("name") == feature_name for f in e.features)]

    def query_by_protocol(self, name: str, version: Optional[str] = None) -> List[RegistryEntry]:
        """Find all runs using a specific protocol (optionally version-pinned).
        
        Essential for comparing protocol versions (e.g., did v2.0 improve
        on v1.0?) and ensuring reproducibility (exact version matching).
        
        Args:
            name: Protocol name (e.g., "oil_authentication")
            version: Semantic version (e.g., "2.1.0") or None for all versions
        
        Returns:
            List of RegistryEntry objects matching criteria (empty if none)
        
        Examples:
            >>> from foodspec.registry import FeatureModelRegistry
            >>> from pathlib import Path
            >>> registry = FeatureModelRegistry(Path("runs.json"))
            >>> all_oil = registry.query_by_protocol("oil_authentication")
            >>> v2_oil = registry.query_by_protocol("oil_authentication", "2.0.0")
            >>> print(f"v2.0 improved by {len(v2_oil)/len(all_oil)*100:.0f}% adoption")
        
        See Also:
            - query_by_feature(): Filter by feature instead
        """
        return [
            e for e in self.entries if e.protocol_name == name and (version is None or e.protocol_version == version)
        ]

    def list_models(self) -> List[str]:
        """List all registered model IDs.
        
        Returns model_id for entries with trained models (excludes analysis-only
        runs). Useful for inventory and finding latest production model.
        
        Returns:
            List of model_id strings (empty if no models registered)
        
        Examples:
            >>> from foodspec.registry import FeatureModelRegistry
            >>> from pathlib import Path
            >>> registry = FeatureModelRegistry(Path("runs.json"))
            >>> models = registry.list_models()
            >>> print(f"Registered models: {len(models)}")
            >>> if models:
            ...     print(f"Latest: {models[-1]}")
        
        See Also:
            - register_model(): Add new model to registry
        """
        return [e.model_id for e in self.entries if e.model_id]
