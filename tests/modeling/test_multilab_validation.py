from __future__ import annotations

import numpy as np

from foodspec.modeling.api import fit_predict
from foodspec.modeling.validation.multilab import multilab_metrics


def _make_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 6))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    labs = np.array(["lab_a"] * 6 + ["lab_b"] * 6)
    return X, y, labs


def test_lolo_scheme_metrics_by_group():
    X, y, labs = _make_data()
    result = fit_predict(X, y, model_name="logreg", scheme="lolo", groups=labs, seed=0, allow_random_cv=True)
    assert "metrics_by_group" in result.diagnostics


def test_multilab_metrics_bundle():
    X, y, labs = _make_data()
    y_pred = y.copy()
    report = multilab_metrics(y, y_pred, labs)
    assert report["lab_count"] == 2
