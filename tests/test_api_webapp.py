import pandas as pd
from fastapi.testclient import TestClient

from webapp.backend.main import app, MODELS_DIR


def test_models_endpoint(tmp_path, monkeypatch):
    client = TestClient(app)
    # create dummy model files
    m = MODELS_DIR / "dummy.json"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    m.write_text("{}")
    resp = client.get("/models")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_predict_endpoint(tmp_path, monkeypatch):
    # Create a fake model by saving empty json/pkl; use monkeypatch to point MODELS_DIR to tmp
    monkeypatch.setenv("FOODSPEC_MODELS_DIR", str(tmp_path))
    from importlib import reload
    from webapp.backend import main
    reload(main)
    mdl_prefix = tmp_path / "toy"
    (mdl_prefix.with_suffix(".json")).write_text("{}")
    import pickle

    with (mdl_prefix.with_suffix(".pkl")).open("wb") as f:
        pickle.dump({"model": None, "scaler": None}, f)

    client = TestClient(main.app)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    files = {"file": ("data.csv", df.to_csv(index=False), "text/csv")}
    resp = client.post("/predict", data={"model_name": "toy"}, files=files)
    # This will fail because model/scaler is None, but endpoint should still respond with 500 or 400
    assert resp.status_code in (400, 500)
