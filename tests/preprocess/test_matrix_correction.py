import numpy as np
import pandas as pd

from foodspec import FoodSpec


def make_synthetic_matrix_dataset(n_per_matrix=10, n_wn=200):
    wn = np.linspace(400, 3000, n_wn)
    # Matrix A baseline +0.3, Matrix B baseline -0.3
    A = np.random.randn(n_per_matrix, n_wn) * 0.02 + 1.0 + 0.3
    B = np.random.randn(n_per_matrix, n_wn) * 0.02 + 1.0 - 0.3
    X = np.vstack([A, B])
    matrix = np.array(["A"] * n_per_matrix + ["B"] * n_per_matrix)
    meta = pd.DataFrame({"matrix_type": matrix})
    return X, wn, meta


def test_apply_matrix_correction_runs_and_records_metrics():
    X, wn, meta = make_synthetic_matrix_dataset()
    fs = FoodSpec(X, wavenumbers=wn, metadata=meta, modality="raman")
    fs.apply_matrix_correction(
        method="adaptive_baseline", scaling="median_mad", domain_adapt=True, matrix_column="matrix_type"
    )
    # Invariants: shape preserved
    assert fs.data.x.shape == X.shape
    # Metrics recorded
    assert any(k.startswith("matrix_correction_") for k in fs.bundle.metrics.keys())
