"""Dataset fingerprinting and manifest generation for reproducibility.

Computes SHA256 hashes of input files and protocol, captures environment
metadata, and generates manifest.json for artifact tracking.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _now_utc_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path, max_mb: int = 200) -> Optional[str]:
    """Compute SHA256 hash of a file, skipping if > max_mb.
    
    Parameters
    ----------
    path : Path
        File path to hash.
    max_mb : int
        Max file size in MB to hash. Returns None if file larger.
    
    Returns
    -------
    Optional[str]
        Hex digest string or None if skipped.
    """
    if not path.exists():
        return None
    
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return None
    
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> Optional[str]:
    """Get current git commit hash (best-effort)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def compute_dataset_fingerprint(csv_path: Path) -> Dict[str, Any]:
    """Compute fingerprint of input CSV.
    
    Includes SHA256 hash, row count, column list, and missingness.
    
    Parameters
    ----------
    csv_path : Path
        Path to input CSV file.
    
    Returns
    -------
    Dict[str, Any]
        Fingerprint with keys: sha256, rows, columns, missing_per_column.
    """
    fingerprint = {
        "sha256": _sha256_file(csv_path),
        "path": str(csv_path),
        "size_bytes": csv_path.stat().st_size if csv_path.exists() else None,
    }
    
    try:
        df = pd.read_csv(csv_path)
        fingerprint["rows"] = len(df)
        fingerprint["columns"] = list(df.columns)
        fingerprint["missing_per_column"] = {
            col: round(df[col].isna().sum() / len(df) * 100, 2)
            for col in df.columns
        }
    except Exception as e:
        fingerprint["read_error"] = str(e)
    
    return fingerprint


def compute_protocol_fingerprint(protocol_path: Path) -> Dict[str, Any]:
    """Compute fingerprint of protocol file.
    
    Parameters
    ----------
    protocol_path : Path
        Path to protocol YAML or JSON file.
    
    Returns
    -------
    Dict[str, Any]
        Fingerprint with keys: sha256, path.
    """
    return {
        "sha256": _sha256_file(protocol_path),
        "path": str(protocol_path),
        "size_bytes": protocol_path.stat().st_size if protocol_path.exists() else None,
    }


@dataclass
class Manifest:
    """Execution manifest for a workflow run.
    
    Captures protocol hash, dataset hash, environment, versions, timestamps,
    and artifacts for full reproducibility and audit trail.
    
    Examples
    --------
    Build and save a manifest::
    
        manifest = Manifest.build(
            protocol_path=Path("protocol.yaml"),
            input_paths=[Path("data.csv")],
            seed=42,
            mode="research",
        )
        manifest.save(Path("runs/run_1/manifest.json"))
    """

    # Fingerprints
    protocol_fingerprint: Dict[str, Any] = field(default_factory=dict)
    dataset_fingerprints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Environment
    foodspec_version: str = ""
    python_version: str = ""
    platform_info: str = ""
    git_commit: Optional[str] = None
    
    # Execution parameters
    seed: Optional[int] = None
    mode: str = "research"
    cli_args: Dict[str, Any] = field(default_factory=dict)
    cli_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: Optional[float] = None
    
    # Artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Contract tracking (Phase 3)
    artifact_contract_version: str = "v3"
    artifact_contract_digest: str = ""
    
    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        protocol_path: Path,
        input_paths: List[Path],
        seed: Optional[int] = None,
        mode: str = "research",
        cli_args: Optional[Dict[str, Any]] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> Manifest:
        """Build manifest from workflow configuration.
        
        Parameters
        ----------
        protocol_path : Path
            Path to protocol file.
        input_paths : List[Path]
            List of input CSV paths.
        seed : Optional[int]
            Random seed used.
        mode : str
            Workflow mode ('research' or 'regulatory').
        cli_args : Optional[Dict]
            Full CLI arguments snapshot.
        cli_overrides : Optional[Dict]
            Fields overridden via CLI.
        
        Returns
        -------
        Manifest
            New manifest instance.
        """
        try:
            from foodspec import __version__
            version = __version__
        except Exception:
            version = "unknown"
        
        return cls(
            protocol_fingerprint=compute_protocol_fingerprint(protocol_path),
            dataset_fingerprints=[compute_dataset_fingerprint(p) for p in input_paths],
            foodspec_version=version,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform_info=platform.platform(),
            git_commit=_git_commit(),
            seed=seed,
            mode=mode,
            cli_args=cli_args or {},
            cli_overrides=cli_overrides or {},
            start_time=_now_utc_iso(),
        )

    def finalize(self) -> None:
        """Mark manifest as complete (end time, duration)."""
        self.end_time = _now_utc_iso()
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
            self.duration_seconds = (end - start).total_seconds()

    def save(self, path: Path) -> None:
        """Save manifest to JSON file.
        
        Parameters
        ----------
        path : Path
            Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict
        data = asdict(self)
        
        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load manifest from JSON file.
        
        Parameters
        ----------
        path : Path
            Input file path.
        
        Returns
        -------
        Manifest
            Loaded manifest instance.
        """
        with path.open("r") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary.
        
        Returns
        -------
        Dict[str, Any]
        """
        return asdict(self)

    def summary_str(self) -> str:
        """Return human-readable summary.
        
        Returns
        -------
        str
        """
        lines = [
            f"Manifest (v{self.foodspec_version})",
            f"  Mode: {self.mode}",
            f"  Seed: {self.seed}",
            f"  Python: {self.python_version}",
            f"  Platform: {self.platform_info}",
            f"  Git: {self.git_commit[:8] if self.git_commit else 'N/A'}",
            f"  Inputs: {len(self.dataset_fingerprints)} file(s)",
            f"  Start: {self.start_time}",
        ]
        if self.duration_seconds:
            lines.append(f"  Duration: {self.duration_seconds:.2f}s")
        return "\n".join(lines)
