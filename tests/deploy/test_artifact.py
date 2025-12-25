from pathlib import Path

import numpy as np

from foodspec.artifact import load_artifact, save_artifact
from foodspec.core.output_bundle import OutputBundle
from foodspec.core.run_record import RunRecord


class DummyModel:
    def predict(self, X):
        return np.array([1] * len(X))

    def predict_proba(self, X):
        return np.tile(np.array([[0.2, 0.8]]), (len(X), 1))


def test_save_and_load_artifact(tmp_path: Path):
    record = RunRecord("test", {"a": 1}, "datahash")
    bundle = OutputBundle(run_record=record)
    bundle.add_artifact("model", DummyModel())
    bundle.add_metrics("cv_metrics", {"acc": 0.9})

    path = tmp_path / "model.foodspec"
    save_artifact(bundle, path, target_grid=np.array([1.0, 2.0]), schema={"label_column": "y"})

    predictor = load_artifact(path)
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = predictor.predict(data)

    assert (res["predictions"] == np.array([1, 1])).all()
    assert res["qc_flags"] is not None
    assert predictor.metadata.schema.get("label_column") == "y"


def test_artifact_version_guard(tmp_path: Path):
    record = RunRecord("test", {"a": 1}, "datahash")
    bundle = OutputBundle(run_record=record)
    bundle.add_artifact("model", DummyModel())
    path = tmp_path / "model.foodspec"
    save_artifact(bundle, path)

    import json
    import zipfile

    with zipfile.ZipFile(path, "a") as zf:
        meta = json.loads(zf.read("metadata.json"))
        meta["foodspec_version"] = "9.9.9"  # simulate future artifact
        zf.writestr("metadata.json", json.dumps(meta))

    import pytest

    with pytest.raises(ValueError):
        load_artifact(path)
    # Allow loading when explicitly unsafe
    predictor = load_artifact(path, allow_incompatible=True)
    assert predictor.metadata.foodspec_version == "9.9.9"
