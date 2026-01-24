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

Hash-based caching for expensive pipeline stages (preprocess, features).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _stable_hash(obj: Any) -> str:
    """Compute stable SHA256 hash of JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    """Cached stage output with metadata."""

    key: str
    stage_name: str
    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.arrays, dict):
            raise TypeError(f"arrays must be dict, got {type(self.arrays)}")
        if not all(isinstance(v, np.ndarray) for v in self.arrays.values()):
            raise TypeError("All array values must be numpy arrays")


class CacheManager:
    """Manage hash-based cache for expensive pipeline stages.

    Cache keys are computed from:
    - Data fingerprint (hash of input data file or array)
    - Stage specification (expanded preprocess steps, feature strategy)
    - Stage name (preprocess, features)
    - Library version (foodspec version for reproducibility)

    Cached outputs are stored as:
    - Arrays: .npz files (compressed numpy archives)
    - Metadata: .json sidecar files

    Examples
    --------
    Cache preprocessed spectra::

        cache = CacheManager(cache_dir=Path(".cache"))
        key = cache.compute_key(
            data_fingerprint="abc123...",
            stage_spec={"recipe": "raman_default"},
            stage_name="preprocess",
            library_version="2.0.0"
        )

        # Check cache
        entry = cache.get(key)
        if entry:
            spectra = entry.arrays["spectra"]
        else:
            # Expensive computation
            spectra = preprocess_data(...)
            cache.put(
                key=key,
                stage_name="preprocess",
                arrays={"spectra": spectra},
                metadata={"shape": spectra.shape, "dtype": str(spectra.dtype)}
            )

    Clear old cache entries::

        cache.clear()
    """

    def __init__(self, cache_dir: Path, enabled: bool = True):
        """Initialize cache manager.

        Parameters
        ----------
        cache_dir : Path
            Directory to store cached outputs.
        enabled : bool, default True
            Enable caching. If False, cache operations become no-ops.
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_key(
        self,
        data_fingerprint: str,
        stage_spec: Dict[str, Any],
        stage_name: str,
        library_version: str,
    ) -> str:
        """Compute stable cache key from inputs.

        Parameters
        ----------
        data_fingerprint : str
            Hash of input data (from RunManifest or computed on-the-fly).
        stage_spec : Dict[str, Any]
            Stage configuration (e.g., preprocess steps, feature strategy).
        stage_name : str
            Stage identifier (preprocess, features).
        library_version : str
            FoodSpec version for cache invalidation on upgrades.

        Returns
        -------
        str
            SHA256 hash serving as cache key.

        Examples
        --------
        >>> cache = CacheManager(Path(".cache"))
        >>> key = cache.compute_key(
        ...     data_fingerprint="abc123",
        ...     stage_spec={"recipe": "raman_default"},
        ...     stage_name="preprocess",
        ...     library_version="2.0.0"
        ... )
        >>> len(key)
        64
        """
        payload = {
            "data_fingerprint": data_fingerprint,
            "stage_spec": stage_spec,
            "stage_name": stage_name,
            "library_version": library_version,
        }
        return _stable_hash(payload)

    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve cached entry by key.

        Parameters
        ----------
        key : str
            Cache key computed by compute_key().

        Returns
        -------
        CacheEntry | None
            Cached entry if exists and valid, otherwise None.

        Examples
        --------
        >>> cache = CacheManager(Path(".cache"))
        >>> entry = cache.get("abc123...")
        >>> if entry:
        ...     spectra = entry.arrays["spectra"]
        """
        if not self.enabled:
            return None

        npz_path = self.cache_dir / f"{key}.npz"
        json_path = self.cache_dir / f"{key}.json"

        if not (npz_path.exists() and json_path.exists()):
            return None

        try:
            # Load arrays
            with np.load(npz_path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}

            # Load metadata
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            stage_name = metadata.get("stage_name", "unknown")

            return CacheEntry(
                key=key,
                stage_name=stage_name,
                arrays=arrays,
                metadata=metadata,
            )
        except Exception:
            # Corrupted cache entry; ignore and recompute
            return None

    def put(
        self,
        key: str,
        stage_name: str,
        arrays: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store stage output in cache.

        Parameters
        ----------
        key : str
            Cache key from compute_key().
        stage_name : str
            Stage identifier (preprocess, features).
        arrays : Dict[str, np.ndarray]
            Named arrays to cache (e.g., {"spectra": X, "wavenumbers": wn}).
        metadata : Dict[str, Any], optional
            Additional metadata to store alongside arrays.

        Examples
        --------
        >>> cache = CacheManager(Path(".cache"))
        >>> spectra = np.random.rand(100, 1000)
        >>> cache.put(
        ...     key="abc123...",
        ...     stage_name="preprocess",
        ...     arrays={"spectra": spectra},
        ...     metadata={"n_samples": 100, "n_features": 1000}
        ... )
        """
        if not self.enabled:
            return

        npz_path = self.cache_dir / f"{key}.npz"
        json_path = self.cache_dir / f"{key}.json"

        # Validate inputs
        if not isinstance(arrays, dict):
            raise TypeError(f"arrays must be dict, got {type(arrays)}")
        if not all(isinstance(v, np.ndarray) for v in arrays.values()):
            raise TypeError("All array values must be numpy arrays")

        # Write arrays
        np.savez_compressed(npz_path, **arrays)

        # Write metadata
        meta = metadata or {}
        meta["stage_name"] = stage_name
        meta["key"] = key
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def exists(self, key: str) -> bool:
        """Check if cache entry exists.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if both .npz and .json files exist.
        """
        if not self.enabled:
            return False
        npz_path = self.cache_dir / f"{key}.npz"
        json_path = self.cache_dir / f"{key}.json"
        return npz_path.exists() and json_path.exists()

    def clear(self) -> int:
        """Clear all cache entries.

        Returns
        -------
        int
            Number of cache entries removed.

        Examples
        --------
        >>> cache = CacheManager(Path(".cache"))
        >>> n_removed = cache.clear()
        >>> print(f"Removed {n_removed} cache entries")
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        count = 0
        for path in self.cache_dir.glob("*.npz"):
            path.unlink()
            count += 1
            # Remove companion json
            json_path = path.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()

        return count

    def size_bytes(self) -> int:
        """Compute total cache size in bytes.

        Returns
        -------
        int
            Total size of all cache files.
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total


__all__ = ["CacheManager", "CacheEntry"]
