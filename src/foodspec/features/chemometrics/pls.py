"""PLS helpers (shim to foodspec.chemometrics.models)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from foodspec.chemometrics.models import make_pls_da, make_pls_regression


def fit_pls_model(
    X: np.ndarray, y: np.ndarray, n_components: int = 10, mode: Literal["regression", "classification"] = "regression"
):
    """Fit a PLS model (regression or PLS-DA classification)."""

    if mode == "classification":
        model = make_pls_da(n_components=n_components)
    else:
        model = make_pls_regression(n_components=n_components)
    return model.fit(X, y)


__all__ = ["fit_pls_model"]
