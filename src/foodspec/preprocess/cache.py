"""Caching and provenance tracking for preprocessing.

Provides:
- Hash-based caching for expensive preprocessing
- Manifest generation with full provenance
- Deterministic behavior with seeding
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from foodspec import __version__ as foodspec_version


def compute_data_hash(X: np.ndarray, wavenumbers: Optional[np.ndarray] = None) -> str:
    """Compute hash of spectral data.

    Parameters
    ----------
    X : np.ndarray
        Spectral intensities (n_samples, n_features).
    wavenumbers : np.ndarray | None
        Wavenumber axis.

    Returns
    -------
    str
        SHA256 hash (first 16 chars).
    """
    hasher = hashlib.sha256()

    # Hash spectral data
    hasher.update(X.tobytes())

    # Hash wavenumbers if provided
    if wavenumbers is not None:
        hasher.update(wavenumbers.tobytes())

    return hasher.hexdigest()[:16]


def compute_recipe_hash(recipe_dict: Dict[str, Any]) -> str:
    """Compute hash of preprocessing recipe.

    Parameters
    ----------
    recipe_dict : Dict[str, Any]
        Recipe configuration dictionary.

    Returns
    -------
    str
        SHA256 hash (first 16 chars).
    """
    # Serialize recipe deterministically
    recipe_json = json.dumps(recipe_dict, sort_keys=True)
    return hashlib.sha256(recipe_json.encode()).hexdigest()[:16]


def compute_cache_key(
    data_hash: str,
    recipe_hash: str,
    seed: Optional[int] = None,
    version: str = foodspec_version,
) -> str:
    """Compute cache key from components.

    Parameters
    ----------
    data_hash : str
        Hash of input data.
    recipe_hash : str
        Hash of preprocessing recipe.
    seed : int | None
        Random seed (if used).
    version : str
        FoodSpec version.

    Returns
    -------
    str
        Combined cache key.
    """
    components = f"{data_hash}:{recipe_hash}:{seed}:{version}"
    return hashlib.sha256(components.encode()).hexdigest()[:16]


class PreprocessCache:
    """Simple file-based cache for preprocessing results."""

    def __init__(self, cache_dir: str | Path):
        """Initialize cache.

        Parameters
        ----------
        cache_dir : str | Path
            Directory for cache storage.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result.

        Parameters
        ----------
        cache_key : str
            Cache key to lookup.

        Returns
        -------
        Dict[str, Any] | None
            Cached result or None if not found.
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"
        meta_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists() or not meta_file.exists():
            return None

        # Load data
        data = np.load(cache_file)
        X_cached = data["X"]
        wavenumbers = data.get("wavenumbers")

        # Load metadata
        with open(meta_file) as f:
            meta = json.load(f)

        return {"X": X_cached, "wavenumbers": wavenumbers, "metadata": meta}

    def put(
        self,
        cache_key: str,
        X: np.ndarray,
        wavenumbers: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store result in cache.

        Parameters
        ----------
        cache_key : str
            Cache key.
        X : np.ndarray
            Preprocessed spectral data.
        wavenumbers : np.ndarray | None
            Wavenumber axis.
        metadata : Dict | None
            Additional metadata to store.
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"
        meta_file = self.cache_dir / f"{cache_key}.json"

        # Save data
        if wavenumbers is not None:
            np.savez_compressed(cache_file, X=X, wavenumbers=wavenumbers)
        else:
            np.savez_compressed(cache_file, X=X)

        # Save metadata
        meta = metadata or {}
        meta["cache_key"] = cache_key
        meta["cached_at"] = datetime.now().isoformat()

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    def clear(self):
        """Clear all cached files."""
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
        for f in self.cache_dir.glob("*.json"):
            f.unlink()


class PreprocessManifest:
    """Provenance tracking for preprocessing runs."""

    def __init__(
        self,
        run_id: str,
        recipe: Dict[str, Any],
        cache_key: str,
        seed: Optional[int] = None,
    ):
        """Initialize manifest.

        Parameters
        ----------
        run_id : str
            Unique run identifier.
        recipe : Dict[str, Any]
            Preprocessing recipe used.
        cache_key : str
            Cache key for this run.
        seed : int | None
            Random seed used.
        """
        self.run_id = run_id
        self.recipe = recipe
        self.cache_key = cache_key
        self.seed = seed
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.statistics: Dict[str, Any] = {}
        self.operators_applied: list[Dict[str, Any]] = []
        self.warnings: list[str] = []

    def record_operator(self, op_name: str, time_ms: float, **kwargs):
        """Record operator execution.

        Parameters
        ----------
        op_name : str
            Operator name.
        time_ms : float
            Execution time in milliseconds.
        **kwargs
            Additional operator-specific metadata.
        """
        self.operators_applied.append({"op": op_name, "time_ms": time_ms, **kwargs})

    def add_warning(self, message: str):
        """Add warning message."""
        self.warnings.append(message)

    def finalize(
        self,
        n_samples_input: int,
        n_samples_output: int,
        n_features: int,
        rejected_spectra: int = 0,
        rejection_reasons: Optional[list[str]] = None,
    ):
        """Finalize manifest with statistics.

        Parameters
        ----------
        n_samples_input : int
            Number of input samples.
        n_samples_output : int
            Number of output samples.
        n_features : int
            Number of features.
        rejected_spectra : int
            Number of rejected spectra.
        rejection_reasons : list[str] | None
            Reasons for rejection.
        """
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        self.statistics = {
            "n_samples_input": n_samples_input,
            "n_samples_output": n_samples_output,
            "n_features": n_features,
            "rejected_spectra": rejected_spectra,
            "rejection_reasons": rejection_reasons or [],
        }

        self.statistics["duration_seconds"] = duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "recipe": self.recipe,
            "cache_key": self.cache_key,
            "seed": self.seed,
            "foodspec_version": foodspec_version,
            "timestamps": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat() if self.end_time else None,
            },
            "statistics": self.statistics,
            "operators_applied": self.operators_applied,
            "warnings": self.warnings,
        }

    def save(self, path: str | Path):
        """Save manifest to JSON file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PreprocessManifest:
        """Load manifest from JSON file.

        Parameters
        ----------
        path : str | Path
            Input file path.

        Returns
        -------
        PreprocessManifest
            Loaded manifest.
        """
        with open(path) as f:
            data = json.load(f)

        manifest = cls(
            run_id=data["run_id"],
            recipe=data["recipe"],
            cache_key=data["cache_key"],
            seed=data.get("seed"),
        )

        manifest.start_time = datetime.fromisoformat(data["timestamps"]["start"])
        if data["timestamps"]["end"]:
            manifest.end_time = datetime.fromisoformat(data["timestamps"]["end"])

        manifest.statistics = data.get("statistics", {})
        manifest.operators_applied = data.get("operators_applied", [])
        manifest.warnings = data.get("warnings", [])

        return manifest


__all__ = [
    "compute_data_hash",
    "compute_recipe_hash",
    "compute_cache_key",
    "PreprocessCache",
    "PreprocessManifest",
]
