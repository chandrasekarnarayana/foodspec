"""Simple model registry for saving/loading trained models with metadata."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib

__all__ = ["ModelMetadata", "save_model", "load_model"]


@dataclass
class ModelMetadata:
    """Metadata describing a saved model artifact."""

    name: str
    version: str
    created_at: str
    foodspec_version: str
    extra: Dict[str, Any]


def _base_paths(path: str | pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    p = pathlib.Path(path)
    return p.with_suffix(".joblib"), p.with_suffix(".json")


def save_model(
    model: Any,
    path: str | pathlib.Path,
    name: str,
    version: str,
    foodspec_version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model and metadata to disk."""

    model_path, meta_path = _base_paths(path)
    now_utc = datetime.now(timezone.utc)
    metadata = ModelMetadata(
        name=name,
        version=version,
        created_at=now_utc.isoformat().replace("+00:00", "Z"),
        foodspec_version=foodspec_version,
        extra=extra or {},
    )
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metadata.__dict__), encoding="utf-8")


def load_model(path: str | pathlib.Path) -> tuple[Any, ModelMetadata]:
    """Load model and associated metadata."""

    model_path, meta_path = _base_paths(path)
    model = joblib.load(model_path)
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = ModelMetadata(**meta_dict)
    return model, metadata
