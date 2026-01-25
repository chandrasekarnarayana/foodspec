import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.qc.engine import compute_health_scores
from foodspec.qc.dataset_qc import check_class_balance


def test_qc_health_scores():
    X = np.random.rand(5, 20)
    wn = np.linspace(400, 1800, 20)
    meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(5)]})
    ds = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")

    result = compute_health_scores(ds)
    assert "health_score" in result.table.columns


def test_qc_dataset_balance():
    df = pd.DataFrame({"label": ["a", "a", "b", "b", "b"]})
    metrics = check_class_balance(df, "label")
    assert "imbalance_ratio" in metrics

