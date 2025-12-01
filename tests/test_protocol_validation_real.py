import pandas as pd
import numpy as np

from foodspec.apps import protocol_validation as pv
from foodspec.core.dataset import FoodSpectrumSet


def _cls_ds():
    wn = np.linspace(1000, 1010, 12)
    X = np.random.default_rng(2).normal(size=(6, wn.size))
    labels = ["c1", "c1", "c2", "c2", "c3", "c3"]
    meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)], "oil_type": labels})
    return FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")


def _mix_ds():
    wn = np.linspace(1000, 1010, 12)
    X = np.random.default_rng(3).normal(size=(6, wn.size))
    meta = pd.DataFrame({"sample_id": [f"m{i}" for i in range(6)], "mixture_fraction_evoo": np.linspace(0, 1, 6)})
    return FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")


def test_protocol_validation_benchmarks_full(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "load_public_mendeley_oils", lambda: _cls_ds())
    monkeypatch.setattr(pv, "load_public_evoo_sunflower_raman", lambda: _mix_ds())
    # patch train_test_split to use larger test_size to satisfy class counts
    from sklearn.model_selection import train_test_split as sk_split

    monkeypatch.setattr(pv, "train_test_split", lambda *args, **kwargs: sk_split(*args, test_size=0.5, stratify=args[1], random_state=0))

    summary = pv.run_protocol_benchmarks(output_dir=tmp_path, random_state=0)
    assert "classification" in summary
    assert "mixture" in summary
