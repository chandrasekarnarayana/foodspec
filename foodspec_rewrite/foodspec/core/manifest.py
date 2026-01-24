"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

RunManifest captures execution metadata and artifacts for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass
class RunManifest:
    """Execution manifest capturing protocol, environment, data, and artifacts.

    Examples
    --------
    Build, save, and reload a manifest::

        manifest = RunManifest.build(
            protocol_snapshot={"version": "2.0.0", "task": {"name": "cls"}},
            data_path=Path("data.csv"),
            seed=123,
            artifacts={"metrics": "metrics.csv"},
        )
        manifest.save(Path("/tmp/run/manifest.json"))
        loaded = RunManifest.load(Path("/tmp/run/manifest.json"))
    """

    protocol_hash: str
    protocol_snapshot: Mapping[str, Any]
    python_version: str
    platform: str
    dependencies: Dict[str, str]
    seed: Optional[int]
    data_fingerprint: str
    start_time: str
    end_time: str
    duration_seconds: float
    artifacts: Dict[str, str]
    warnings: List[str] = field(default_factory=list)
    cache_hits: List[str] = field(default_factory=list)
    cache_misses: List[str] = field(default_factory=list)
    hyperparameters_per_fold: List[Dict[str, Any]] = field(default_factory=list)

    # Construction helpers
    @classmethod
    def build(
        cls,
        protocol_snapshot: Mapping[str, Any],
        data_path: Optional[Path],
        seed: Optional[int],
        artifacts: Dict[str, str],
        warnings: Optional[List[str]] = None,
        cache_hits: Optional[List[str]] = None,
        cache_misses: Optional[List[str]] = None,
        hyperparameters_per_fold: Optional[List[Dict[str, Any]]] = None,
        dependencies: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> "RunManifest":
        start = start_time or _now_utc()
        end = end_time or _now_utc()
        protocol_hash = cls.compute_protocol_hash(protocol_snapshot)
        data_fingerprint = cls.compute_data_fingerprint(data_path) if data_path else ""

        deps = dependencies or {}
        warning_list = warnings or []
        cache_hit_list = cache_hits or []
        cache_miss_list = cache_misses or []
        hyperparams_list = hyperparameters_per_fold or []

        return cls(
            protocol_hash=protocol_hash,
            protocol_snapshot=protocol_snapshot,
            python_version=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()}",
            dependencies=deps,
            seed=seed,
            data_fingerprint=data_fingerprint,
            start_time=_iso(start),
            end_time=_iso(end),
            duration_seconds=max((end - start).total_seconds(), 0.0),
            artifacts=artifacts,
            warnings=warning_list,
            cache_hits=cache_hit_list,
            cache_misses=cache_miss_list,
            hyperparameters_per_fold=hyperparams_list,
        )

    @staticmethod
    def compute_protocol_hash(protocol_snapshot: Mapping[str, Any]) -> str:
        """Compute a deterministic hash for the protocol snapshot."""

        serialized = json.dumps(protocol_snapshot, sort_keys=True, default=str).encode()
        return _sha256_bytes(serialized)

    @staticmethod
    def compute_data_fingerprint(path: Path) -> str:
        """Hash file contents for provenance."""

        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # Persistence
    def save(self, path: Path) -> None:
        """Save manifest to JSON, creating parent directories."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        """Load manifest from JSON."""

        payload = json.loads(path.read_text())
        return cls(**payload)


__all__ = ["RunManifest"]
