import numpy as np
import pandas as pd

from foodspec import FoodSpec


def make_governance_dataset(n=60, n_wn=100):
    wn = np.linspace(400, 3000, n_wn)
    # Two classes with mild imbalance
    n_a, n_b = 40, 20
    X = np.random.randn(n, n_wn) * 0.03 + 1.0
    labels = ["A"] * n_a + ["B"] * n_b
    batches = ["batch1"] * 30 + ["batch2"] * 30
    # Replicates: few duplicates
    sample_ids = [f"s{i}" for i in range(n)]
    for i in range(0, n, 10):
        if i + 1 < n:
            sample_ids[i + 1] = sample_ids[i]
    meta = pd.DataFrame({"label": labels, "batch": batches, "sample_id": sample_ids})
    return X, wn, meta


def test_data_governance_suite():
    X, wn, meta = make_governance_dataset()
    fs = FoodSpec(X, wavenumbers=wn, metadata=meta, modality="raman")
    summary = fs.summarize_dataset(label_column="label")
    assert "spectral_quality" in summary and "metadata_completeness" in summary
    balance = fs.check_class_balance(label_column="label")
    assert "imbalance_ratio" in balance
    consistency = fs.assess_replicate_consistency(replicate_column="sample_id")
    assert "median_cv" in consistency
    leakage = fs.detect_leakage(label_column="label", batch_column="batch", replicate_column="sample_id")
    assert "overall_risk" in leakage
    readiness = fs.compute_readiness_score(label_column="label", batch_column="batch", replicate_column="sample_id")
    assert 0 <= readiness["overall_score"] <= 100
