from __future__ import annotations

import numpy as np

from foodspec.data_objects.spectra_set import FoodSpectrumSet
from foodspec.qc.engine import detect_outliers


def _make_ds() -> FoodSpectrumSet:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 20))
    return FoodSpectrumSet(x=X, wavenumbers=np.arange(X.shape[1]), metadata=None, modality="raman")


def test_ocsvm_outliers():
    ds = _make_ds()
    result = detect_outliers(ds, method="ocsvm", nu=0.2)
    assert result.labels.shape[0] == ds.x.shape[0]
    assert result.scores.shape[0] == ds.x.shape[0]


def test_elliptic_envelope_outliers():
    ds = _make_ds()
    result = detect_outliers(ds, method="elliptic_envelope", contamination=0.2)
    assert result.labels.shape[0] == ds.x.shape[0]
    assert result.scores.shape[0] == ds.x.shape[0]
