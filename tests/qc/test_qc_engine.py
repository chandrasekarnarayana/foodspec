import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.qc.engine import (
    compute_health_scores,
    detect_drift,
    detect_outliers,
    generate_qc_report,
)


def _toy_ds(n: int = 8, m: int = 40) -> FoodSpectrumSet:
    x_axis = np.linspace(0, 10, m)
    spectra = []
    for i in range(n):
        spectra.append(np.sin(x_axis) + 0.05 * np.random.randn(m) + 0.1 * i)
    spectra = np.vstack(spectra)
    meta = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(n)], "batch_id": ["b1"] * (n // 2) + ["b2"] * (n - n // 2)}
    )
    return FoodSpectrumSet(x=spectra, wavenumbers=x_axis, metadata=meta, modality="raman", batch_col="batch_id")


def test_health_scores_produce_table_and_bounds():
    ds = _toy_ds()
    res = compute_health_scores(ds)
    assert not res.table.empty
    assert (res.table["health_score"] <= 1).all()
    assert (res.table["health_score"] >= 0).all()


def test_outlier_detection_methods():
    ds = _toy_ds()
    for method in ["robust_z", "mahalanobis"]:
        res = detect_outliers(ds, method=method)
        assert res.labels.shape[0] == len(ds)


def test_drift_detection_runs():
    ds = _toy_ds()
    res = detect_drift(ds, method="pca_cusum")
    assert res.drift_score >= 0


def test_qc_report_recommendation():
    ds = _toy_ds()
    report = generate_qc_report(ds)
    assert report.recommendations in {"ok", "recollect", "recalibrate", "exclude"}
