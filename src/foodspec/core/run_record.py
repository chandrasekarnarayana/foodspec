"""Run record for provenance tracking: config, dataset, step hashes, environment versions."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RunRecord:
    """Immutable record of a workflow execution with full provenance.

    Tracks configuration, dataset hash, step history, environment, timing, and user info.

    Args:
        workflow_name (str): Name of the workflow (e.g., "oil_authentication").
        config (dict): Configuration parameters.
        dataset_hash (str): SHA256 hash of input dataset.
        environment (dict | None): Environment info (Python version, package versions, etc.).
        step_records (list[dict] | None): Step records: {"name","hash","timestamp","error"}.
        user (str | None): User who ran the workflow.
        notes (str | None): Freeform notes.

    Attributes:
        workflow_name: Name of the workflow.
        config: Configuration parameters.
        config_hash: SHA256 hash of config.
        dataset_hash: SHA256 hash of input dataset.
        environment: Environment info (Python version, packages, etc.).
        step_records: Step records with name, hash, timestamp, error.
        user: User who ran the workflow.
        notes: Freeform notes.
        timestamp: ISO 8601 timestamp (UTC).
        run_id: Unique run identifier.
    """

    workflow_name: str
    config: Dict[str, Any]
    dataset_hash: str
    environment: Dict[str, Any] = field(default_factory=dict)
    step_records: List[Dict[str, Any]] = field(default_factory=list)
    random_seeds: Dict[str, Any] = field(default_factory=dict)
    output_paths: List[str] = field(default_factory=list)
    user: Optional[str] = None
    notes: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        """Finalize and validate run record."""
        if not self.environment:
            self.environment = _capture_environment()
        if not self.random_seeds:
            self.random_seeds = _capture_seeds()
        if not self.user:
            self.user = os.getenv("USER", "unknown")

    @property
    def config_hash(self) -> str:
        """Hash of configuration.

        Returns:
            str: First 8 hex chars of SHA256 over config JSON.
        """
        config_str = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

    @property
    def run_id(self) -> str:
        """Unique run identifier combining workflow name and timestamp.

        Returns:
            str: Deterministic identifier for the run.
        """
        ts = self.timestamp.replace(":", "").replace("-", "").replace(".", "")[:12]
        return f"{self.workflow_name}_{ts}"

    @property
    def combined_hash(self) -> str:
        """Combined hash of config + dataset + all steps.

        Returns:
            str: First 8 hex chars of combined SHA256.
        """
        combined_str = f"{self.config_hash}_{self.dataset_hash}_{'_'.join(s['hash'] for s in self.step_records)}"
        return hashlib.sha256(combined_str.encode()).hexdigest()[:8]

    def add_step(
        self,
        name: str,
        step_hash: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a workflow step.

        Args:
            name (str): Step name (e.g., "baseline_correction").
            step_hash (str): Hash of step output or configuration.
            error (str | None): Error message if step failed.
            metadata (dict | None): Additional metadata for this step.
        """
        record = {
            "name": name,
            "hash": step_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "metadata": metadata or {},
        }
        self.step_records.append(record)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            dict: JSON-serializable representation of the run record.
        """
        return {
            "workflow_name": self.workflow_name,
            "run_id": self.run_id,
            "config": self.config,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "environment": self.environment,
            "step_records": self.step_records,
            "random_seeds": self.random_seeds,
            "output_paths": self.output_paths,
            "user": self.user,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "combined_hash": self.combined_hash,
        }

    def to_json(self, path: Path | str) -> Path:
        """Write to JSON file.

        Args:
            path (Path | str): Output file path.

        Returns:
            Path: Path to written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: Path | str) -> RunRecord:
        """Load from JSON file.

        Args:
            path (Path | str): Input file path.

        Returns:
            RunRecord: Deserialized record.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            workflow_name=data["workflow_name"],
            config=data["config"],
            dataset_hash=data["dataset_hash"],
            environment=data.get("environment", {}),
            step_records=data.get("step_records", []),
            random_seeds=data.get("random_seeds", {}),
            output_paths=data.get("output_paths", []),
            user=data.get("user"),
            notes=data.get("notes"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    def add_output_path(self, path: str | Path) -> None:
        """Record an output path for the run (e.g., exported bundle location).

        Args:
            path (str | Path): Output directory/file path.
        """

        self.output_paths.append(str(Path(path)))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RunRecord({self.workflow_name}, run_id={self.run_id}, "
            f"steps={len(self.step_records)}, config_hash={self.config_hash})"
        )


def _capture_environment() -> Dict[str, Any]:
    """Capture environment info: Python version, platform, package versions.

    Returns:
        dict: Environment information.
    """
    import sys

    env = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "hostname": os.getenv("HOSTNAME", "unknown"),
    }

    # Capture key package versions
    try:
        import foodspec

        env["foodspec_version"] = getattr(foodspec, "__version__", "unknown")
    except ImportError:
        pass

    for pkg in ["numpy", "pandas", "sklearn", "scipy", "statsmodels"]:
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return env


def _capture_seeds() -> Dict[str, Any]:
    """Capture random seeds across common libraries for reproducibility.

    Returns:
        dict: Seeds for Python random, NumPy, optional Torch, and hash seed.
    """

    seeds: Dict[str, Any] = {}
    try:
        seeds["python_random_seed"] = random.getstate()[1][0]
    except Exception:
        pass

    try:
        seeds["numpy_seed"] = int(np.random.get_state()[1][0])
    except Exception:
        pass

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is not None:
        seeds["PYTHONHASHSEED"] = hash_seed

    try:  # Optional dependency
        import torch

        seeds["torch_seed"] = int(torch.initial_seed())
    except Exception:
        pass

    return seeds


def _hash_path(path: Path | str) -> str:
    """Compute a stable hash for a file or directory path.

    Args:
        path (Path | str): File or directory to hash.

    Returns:
        str: First 8 hex chars of SHA256 over file contents or directory tree.

    Raises:
        FileNotFoundError: If the path does not exist.
    """

    p = Path(path)
    if p.is_file():
        return _hash_file(p)
    if p.is_dir():
        master = hashlib.sha256()
        for file in sorted(p.rglob("*")):
            if file.is_file():
                master.update(str(file.relative_to(p)).encode())
                master.update(_hash_file_digest(file).digest())
        return master.hexdigest()[:8]
    raise FileNotFoundError(f"Path not found: {path}")


def _hash_file_digest(path: Path, chunk_size: int = 8192) -> hashlib._Hash:
    """Compute SHA256 digest object for a file (streamed).

    Args:
        path (Path): File path.
        chunk_size (int): Chunk size for streaming.

    Returns:
        hashlib._Hash: Digest object.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest


def _hash_file(path: Path | str) -> str:
    """Hash a single file and return first 8 characters of SHA256.

    Args:
        path (Path | str): File path.

    Returns:
        str: First 8 hex chars of the file hash.
    """

    return _hash_file_digest(Path(path)).hexdigest()[:8]


def _hash_data(data: np.ndarray | pd.DataFrame) -> str:
    """Compute hash of data array or DataFrame.

    Args:
        data (np.ndarray | pd.DataFrame): Input data to hash.

    Returns:
        str: SHA256 hash (first 8 characters).
    """
    if isinstance(data, pd.DataFrame):
        data_bytes = pd.util.hash_pandas_object(data, index=True).values.tobytes()
    else:
        data = np.asarray(data)
        data_bytes = data.tobytes()

    return hashlib.sha256(data_bytes).hexdigest()[:8]
