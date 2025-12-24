import numpy as np
import pandas as pd
from foodspec import FoodSpec

def test_apply_calibration_transfer_basic():
    n_std, n_wn = 20, 120
    wn = np.linspace(400, 3000, n_wn)
    # Source standards: baseline 1.0
    source = np.random.randn(n_std, n_wn) * 0.02 + 1.0
    # Target standards: baseline 1.1
    target = source * 1.02 + 0.05  # introduce small linear difference
    # Production target data
    X = np.random.randn(50, n_wn) * 0.02 + 1.05
    meta = pd.DataFrame({"label": ["classA"] * 50})
    fs = FoodSpec(X, wavenumbers=wn, metadata=meta, modality="raman")
    fs.apply_calibration_transfer(source_standards=source, target_standards=target, method="ds", alpha=1.0)
    # Shape preserved
    assert fs.data.x.shape == X.shape
    # Metrics recorded
    assert any(k.startswith("calibration_transfer_") for k in fs.bundle.metrics.keys())
