"""Model export helpers for deployment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib


@dataclass
class PipelineBundle:
    """Loaded pipeline bundle with optional metadata."""

    model: Any
    metadata: Dict[str, Any]

    def predict(self, X):
        return self.model.predict(X)


def save_pipeline(model: Any, path: Path | str, *, metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Serialize a fitted pipeline/model with optional metadata."""
    payload = {"model": model, "metadata": metadata or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def load_pipeline(path: Path | str) -> PipelineBundle:
    """Load a pipeline bundle saved by save_pipeline()."""
    path = Path(path)
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return PipelineBundle(model=payload["model"], metadata=payload.get("metadata", {}))
    return PipelineBundle(model=payload, metadata={})


def export_onnx(
    model: Any,
    path: Path | str,
    *,
    input_dim: int,
    input_name: str = "input",
    opset: int = 15,
) -> Path:
    """Export a fitted scikit-learn model to ONNX."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("skl2onnx is required for ONNX export. Install foodspec[deploy].") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    initial_types = [(input_name, FloatTensorType([None, int(input_dim)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=opset)
    path.write_bytes(onnx_model.SerializeToString())
    return path


def export_pmml(
    model: Any,
    path: Path | str,
    *,
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = "target",
) -> Path:
    """Export a fitted scikit-learn model to PMML."""
    try:
        from sklearn2pmml import sklearn2pmml
        from sklearn2pmml.pipeline import PMMLPipeline
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("sklearn2pmml is required for PMML export. Install foodspec[deploy].") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None:
            feature_names = [f"x{i}" for i in range(int(n_features))]
        else:
            feature_names = []

    pipeline = PMMLPipeline([("model", model)])
    pipeline.active_fields = list(feature_names) if feature_names else None
    pipeline.target_fields = [target_name]
    sklearn2pmml(pipeline, str(path), with_repr=True)
    return path


__all__ = ["PipelineBundle", "save_pipeline", "load_pipeline", "export_onnx", "export_pmml"]
