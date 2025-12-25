import numpy as np
import pandas as pd
from foodspec.core.time import TimeSpectrumSet
from foodspec.workflows.shelf_life import estimate_remaining_shelf_life


def test_estimate_remaining_shelf_life_linear():
    # Two entities with y = 0.5 * t (+ offset for B)
    t_points = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    X = []
    meta_rows = []
    wn = np.linspace(800, 1800, 21)
    for ent in ["A", "B"]:
        for t in t_points:
            X.append(np.zeros_like(wn))
            meta_rows.append({
                "sample_id": ent,
                "time": t,
                "deg": 0.5 * t + (0.1 if ent == "B" else 0.0)
            })
    X = np.vstack(X)
    meta = pd.DataFrame(meta_rows)
    ds = TimeSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman", time_col="time", entity_col="sample_id")

    out = estimate_remaining_shelf_life(ds, value_col="deg", threshold=2.0)
    # A: 0.5*t = 2 -> t*=4; B: 0.5*t + 0.1 = 2 -> t* = 3.8
    t_star = dict(zip(out["entity"], out["t_star"]))
    assert np.isclose(t_star["A"], 4.0, atol=1e-8)
    assert np.isclose(t_star["B"], 3.8, atol=1e-8)
    # CI should be finite numbers
    assert np.isfinite(out["ci_low"]).all()
    assert np.isfinite(out["ci_high"]).all()
