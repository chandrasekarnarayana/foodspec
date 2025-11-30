"""Quality control / novelty detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from foodspec.core.dataset import FoodSpectrumSet

__all__ = ["QCResult", "train_qc_model", "apply_qc_model"]


@dataclass
class QCResult:
    scores: pd.Series
    labels_pred: pd.Series
    threshold: float
    model: Any
    metadata: pd.DataFrame


def train_qc_model(
    spectra: FoodSpectrumSet,
    train_mask: Optional[pd.Series] = None,
    model_type: str = "oneclass_svm",
    **kwargs: Any,
) -> Any:
    """Train a novelty detection model on authentic spectra."""

    X = spectra.x
    if train_mask is not None:
        train_idx = train_mask.to_numpy()
        X = X[train_idx]

    name = model_type.lower()
    if name == "oneclass_svm":
        gamma = kwargs.pop("gamma", "scale")
        nu = kwargs.pop("nu", 0.05)
        model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu, **kwargs)
    elif name == "isolation_forest":
        model = IsolationForest(contamination=0.05, random_state=0, **kwargs)
    else:
        raise ValueError("model_type must be 'oneclass_svm' or 'isolation_forest'.")

    model.fit(X)
    return model


def apply_qc_model(
    spectra: FoodSpectrumSet,
    model: Any,
    threshold: Optional[float] = None,
    higher_score_is_more_normal: bool = True,
    metadata: Optional[pd.DataFrame] = None,
) -> QCResult:
    """Score spectra with a novelty model and produce QC labels."""

    X = spectra.x
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "score_samples"):
        scores = model.score_samples(X)
    else:
        raise ValueError("Model must implement decision_function or score_samples.")

    scores_ser = pd.Series(scores, index=spectra.metadata.index)

    if threshold is None:
        if isinstance(model, OneClassSVM):
            threshold = float(np.quantile(scores, 0.2))
        else:
            threshold = float(np.median(scores))
    if higher_score_is_more_normal:
        labels_arr = np.where(scores >= threshold, "authentic", "suspect")
    else:
        labels_arr = np.where(scores <= threshold, "authentic", "suspect")
    labels_ser = pd.Series(labels_arr, index=spectra.metadata.index)

    meta = metadata.copy() if metadata is not None else spectra.metadata.copy()

    return QCResult(
        scores=scores_ser,
        labels_pred=labels_ser,
        threshold=threshold,
        model=model,
        metadata=meta,
    )
