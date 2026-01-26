from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from foodspec.deploy.export import export_onnx, export_pmml, load_pipeline, save_pipeline


def _fit_model():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression(max_iter=200).fit(X, y)
    return model, X


def test_save_load_pipeline(tmp_path: Path) -> None:
    model, X = _fit_model()
    path = tmp_path / "model.joblib"
    save_pipeline(model, path, metadata={"label": "demo"})
    bundle = load_pipeline(path)
    preds = bundle.model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert bundle.metadata["label"] == "demo"


def test_export_onnx_optional(tmp_path: Path) -> None:
    pytest.importorskip("skl2onnx")
    model, _ = _fit_model()
    path = tmp_path / "model.onnx"
    export_onnx(model, path, input_dim=2)
    assert path.exists()


def test_export_pmml_optional(tmp_path: Path) -> None:
    pytest.importorskip("sklearn2pmml")
    model, _ = _fit_model()
    path = tmp_path / "model.pmml"
    export_pmml(model, path, feature_names=["x0", "x1"], target_name="label")
    assert path.exists()
