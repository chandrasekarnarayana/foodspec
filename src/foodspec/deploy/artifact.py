"""Single-file deployment artifacts (.foodspec) for FoodSpec models."""
from __future__ import annotations


import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np

from foodspec.core.output_bundle import OutputBundle
from foodspec.core.run_record import RunRecord
from foodspec.data.libraries import load_library

try:  # Avoid circular import risk; __version__ is defined before artifact import in __init__
    from foodspec import __version__ as _FOODSPEC_VERSION  # type: ignore
except Exception:  # pragma: no cover - fallback for partial installs
    _FOODSPEC_VERSION = None


ARTIFACT_SCHEMA_VERSION = "1.0"


def _parse_semver(version_str: Optional[str]) -> Tuple[int, int, int]:
    if not version_str:
        return (0, 0, 0)
    try:
        parts = [int(p) for p in str(version_str).split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])  # type: ignore
    except Exception:
        return (0, 0, 0)


def _ensure_compatible(saved: Optional[str], current: Optional[str], *, allow_future: bool) -> None:
    if saved is None or current is None:
        return
    s_major, s_minor, _ = _parse_semver(saved)
    c_major, c_minor, _ = _parse_semver(current)
    if s_major != c_major:
        raise ValueError(f"Artifact built with incompatible major version {saved}; current is {current}.")
    if not allow_future and (s_minor > c_minor):
        raise ValueError(f"Artifact built with newer minor version {saved}; current is {current}.")
    # Patch-level differences are allowed


@dataclass
class ArtifactMetadata:
    pipeline_steps: List[Dict[str, Any]]
    schema: Dict[str, Any]
    feature_specs: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    environment: Dict[str, Any]
    target_grid_present: bool
    foodspec_version: str | None
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_steps": self.pipeline_steps,
            "schema": self.schema,
            "feature_specs": self.feature_specs,
            "metrics": self.metrics,
            "environment": self.environment,
            "target_grid_present": self.target_grid_present,
            "foodspec_version": self.foodspec_version,
            "artifact_schema_version": self.artifact_schema_version,
        }


class Predictor:
    """Lightweight predictor restored from a .foodspec artifact."""

    def __init__(
        self, model: Any, metadata: ArtifactMetadata, run_record: RunRecord, target_grid: Optional[np.ndarray]
    ):
        self.model = model
        self.metadata = metadata
        self.run_record = run_record
        self.target_grid = target_grid

    def predict(self, data: Any) -> Dict[str, Any]:
        X = _extract_features(data)
        preds = self.model.predict(X)
        uncertainty = None
        qc_flags: Optional[np.ndarray] = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)
                max_conf = np.max(proba, axis=1)
                uncertainty = 1.0 - max_conf
                qc_flags = uncertainty < 0.4
            except Exception:
                pass
        return {
            "predictions": preds,
            "uncertainty": uncertainty,
            "qc_flags": qc_flags,
        }


def save_artifact(
    bundle: OutputBundle,
    path: Path | str,
    *,
    target_grid: Optional[np.ndarray] = None,
    feature_specs: Optional[Iterable[Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create a single-file artifact containing model + provenance + schema."""

    if "model" not in bundle.artifacts:
        raise ValueError("bundle.artifacts must include a 'model' entry")

    run_record = bundle.run_record
    model = bundle.artifacts["model"]
    feature_specs_list = [getattr(fs, "__dict__", dict(fs)) for fs in (feature_specs or [])]
    metrics_serializable = OutputBundle._make_serializable(bundle.metrics)  # type: ignore

    current_version = bundle.run_record.environment.get("foodspec_version") or _FOODSPEC_VERSION
    if current_version:
        bundle.run_record.environment["foodspec_version"] = current_version

    meta = ArtifactMetadata(
        pipeline_steps=run_record.step_records,
        schema=schema or {},
        feature_specs=feature_specs_list,
        metrics=metrics_serializable,
        environment=run_record.environment,
        target_grid_present=target_grid is not None,
        foodspec_version=current_version,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # model
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        zf.writestr("model.joblib", model_buffer.getvalue())

        # run record
        zf.writestr("run_record.json", json.dumps(run_record.to_dict(), indent=2))

        # metadata
        zf.writestr("metadata.json", json.dumps(meta.to_dict(), indent=2))

        # target grid
        if target_grid is not None:
            grid_buffer = io.BytesIO()
            np.save(grid_buffer, np.asarray(target_grid))
            zf.writestr("target_grid.npy", grid_buffer.getvalue())

    return path


def load_artifact(path: Path | str, *, allow_incompatible: bool = False) -> Predictor:
    """Load a .foodspec artifact and return a Predictor.

    Set allow_incompatible=True to bypass version guards (not recommended for production).
    """

    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        model_bytes = zf.read("model.joblib")
        model = joblib.load(io.BytesIO(model_bytes))
        run_record = _load_run_record(zf.read("run_record.json"))
        meta_dict = json.loads(zf.read("metadata.json").decode())
        metadata = ArtifactMetadata(**{"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, **meta_dict})
        if not allow_incompatible:
            _ensure_compatible(metadata.artifact_schema_version, ARTIFACT_SCHEMA_VERSION, allow_future=False)
            _ensure_compatible(metadata.foodspec_version, _FOODSPEC_VERSION, allow_future=False)
        target_grid = None
        if "target_grid.npy" in zf.namelist():
            target_grid = np.load(io.BytesIO(zf.read("target_grid.npy")))

    return Predictor(model=model, metadata=metadata, run_record=run_record, target_grid=target_grid)


def _load_run_record(raw: bytes) -> RunRecord:
    data = json.loads(raw.decode())
    return RunRecord(
        workflow_name=data.get("workflow_name", "artifact"),
        config=data.get("config", {}),
        dataset_hash=data.get("dataset_hash", "unknown"),
        environment=data.get("environment", {}),
        step_records=data.get("step_records", []),
        random_seeds=data.get("random_seeds", {}),
        output_paths=data.get("output_paths", []),
        user=data.get("user"),
        notes=data.get("notes"),
        timestamp=data.get("timestamp"),
    )


def _extract_features(data: Any) -> Any:
    """Coerce input into model-ready feature matrix."""

    # FoodSpectrumSet support
    if hasattr(data, "x"):
        return data.x
    if isinstance(data, (str, Path)):
        ds = load_library(data)
        return ds.x
    if isinstance(data, (list, tuple)):
        return np.asarray(data)
    if isinstance(data, np.ndarray):
        return data
    raise ValueError("Unsupported data type for prediction; pass FoodSpectrumSet, ndarray, or compatible object.")


__all__ = ["save_artifact", "load_artifact", "Predictor"]
