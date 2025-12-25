import numpy as np
import pandas as pd

from foodspec import FoodSpec


def make_synthetic_time_series(n_samples=30, n_wn=150):
    wn = np.linspace(400, 3000, n_wn)
    # Create spectral intensity with slight trend over time via index bands
    X = np.random.randn(n_samples, n_wn) * 0.05 + 1.0
    time = np.linspace(0, 48, n_samples)
    # Add a mild increase near 840 cm-1 region
    idx_840 = np.argmin(np.abs(wn - 840))
    X[:, idx_840] += (time / 48.0) * 0.2
    meta = pd.DataFrame({"time_hours": time})
    return X, wn, meta


def test_analyze_heating_trajectory_basic():
    X, wn, meta = make_synthetic_time_series()
    fs = FoodSpec(X, wavenumbers=wn, metadata=meta, modality="raman")
    res = fs.analyze_heating_trajectory(time_column="time_hours", indices=["pi", "tfc"], classify_stages=False, estimate_shelf_life=False)
    assert "trajectory" in res
    # Bundle should have heating trajectory metrics recorded
    assert "heating_trajectory" in fs.bundle.metrics
