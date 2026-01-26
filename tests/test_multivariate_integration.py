from __future__ import annotations

import numpy as np
import pandas as pd

from foodspec.modeling.api import fit_predict
from foodspec.multivariate.pca import PCAComponent
from foodspec.multivariate.stats import HotellingT2Component, MANOVAComponent
from foodspec.protocol.config import ProtocolConfig
from foodspec.protocol.steps.multivariate import MultivariateAnalysisStep


def test_pca_component_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    res = PCAComponent(n_components=2, random_state=0).fit_transform(X)
    assert res.scores.shape == (12, 2)
    assert res.components is not None and res.components.shape[0] == 2


def test_hotelling_and_manova_outputs():
    rng = np.random.default_rng(0)
    x1 = rng.normal(loc=0.0, size=(8, 3))
    x2 = rng.normal(loc=1.0, size=(8, 3))
    X = np.vstack([x1, x2])
    y = np.array([0] * 8 + [1] * 8)
    t2 = HotellingT2Component().fit_transform(X, y)
    assert t2.scores.shape == (1, 1)
    mv = MANOVAComponent().fit_transform(X, y)
    assert "wilks_lambda" in mv.metadata


def test_multivariate_step_produces_artifacts():
    df = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [2, 1, 2, 1], "class": [0, 0, 1, 1]})
    ctx = {"data": df, "tables": {}, "logs": [], "metadata": {}, "figures": {}, "config": ProtocolConfig(name="demo")}
    step = MultivariateAnalysisStep({"method": "pca", "params": {"n_components": 2}})
    step.run(ctx)
    assert "multivariate_scores" in ctx["tables"]
    assert not ctx["tables"]["multivariate_scores"].empty
    assert "multivariate_qc" in ctx["tables"]


def test_fit_predict_with_embedding_pca():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.array([0] * 20 + [1] * 20)
    result = fit_predict(
        X,
        y,
        model_name="logreg",
        scheme="nested",
        seed=0,
        outcome_type="classification",
        embedding={"method": "pca", "params": {"n_components": 3, "random_state": 0}},
    )
    assert result.metrics
    assert result.diagnostics.get("embedding", {}).get("n_components") == 3
