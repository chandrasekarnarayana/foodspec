"""Multi-modal fusion strategies for combining predictions and features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np


@dataclass
class LateFusionResult:
    """Result of late fusion combining features from multiple modalities."""

    X_fused: np.ndarray
    feature_names: List[str]
    modality_boundaries: Dict[str, tuple[int, int]]


def late_fusion_concat(
    feature_dict: Dict[str, np.ndarray],
    modality_order: Optional[Sequence[str]] = None,
) -> LateFusionResult:
    """Concatenate features from multiple modalities.

    Parameters
    ----------
    feature_dict : Dict[str, np.ndarray]
        Mapping from modality name to feature matrix (n_samples, n_features).
    modality_order : Optional[Sequence[str]]
        Order of modalities for concatenation. If None, uses sorted keys.

    Returns
    -------
    LateFusionResult
        Fused feature matrix with metadata.
    """
    if not feature_dict:
        raise ValueError("feature_dict must contain at least one modality.")
    if modality_order is None:
        # Preserve dict insertion order (Python 3.7+)
        modality_order = list(feature_dict.keys())
    n_samples = None
    for modality in modality_order:
        if modality not in feature_dict:
            raise ValueError(f"Modality '{modality}' not found in feature_dict.")
        X = feature_dict[modality]
        if X.ndim != 2:
            raise ValueError(f"Feature matrix for {modality} must be 2D.")
        if n_samples is None:
            n_samples = X.shape[0]
        elif X.shape[0] != n_samples:
            raise ValueError(
                f"Feature matrix for {modality} has {X.shape[0]} samples, "
                f"expected {n_samples}."
            )
    matrices: List[np.ndarray] = []
    boundaries: Dict[str, tuple[int, int]] = {}
    names: List[str] = []
    offset = 0
    for modality in modality_order:
        X = feature_dict[modality]
        n_feat = X.shape[1]
        matrices.append(X)
        boundaries[modality] = (offset, offset + n_feat)
        names.extend([f"{modality}_f{i}" for i in range(n_feat)])
        offset += n_feat
    X_fused = np.hstack(matrices)
    return LateFusionResult(
        X_fused=X_fused, feature_names=names, modality_boundaries=boundaries
    )


@dataclass
class DecisionFusionResult:
    """Result of decision fusion combining predictions from multiple models."""

    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    method: str


def decision_fusion_vote(
    predictions_dict: Dict[str, np.ndarray],
    strategy: Literal["majority", "unanimous"] = "majority",
) -> DecisionFusionResult:
    """Combine class predictions via voting.

    Parameters
    ----------
    predictions_dict : Dict[str, np.ndarray]
        Mapping from modality to 1D array of class predictions.
    strategy : Literal["majority", "unanimous"]
        - "majority": most frequent prediction wins.
        - "unanimous": only assign if all modalities agree.

    Returns
    -------
    DecisionFusionResult
        Fused predictions.
    """
    if not predictions_dict:
        raise ValueError("predictions_dict must contain at least one modality.")
    pred_arrays = list(predictions_dict.values())
    n_samples = pred_arrays[0].shape[0]
    for arr in pred_arrays:
        if arr.shape[0] != n_samples:
            raise ValueError("All prediction arrays must have the same length.")
    stacked = np.vstack(pred_arrays).T  # shape (n_samples, n_modalities)
    if strategy == "majority":
        from scipy.stats import mode
        fused, _ = mode(stacked, axis=1, keepdims=False)
        fused = fused.ravel()
    elif strategy == "unanimous":
        fused = np.full(n_samples, -1, dtype=int)
        for i in range(n_samples):
            row = stacked[i]
            if np.all(row == row[0]):
                fused[i] = row[0]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")
    return DecisionFusionResult(predictions=fused, probabilities=None, method=strategy)


def decision_fusion_weighted(
    proba_dict: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> DecisionFusionResult:
    """Combine class probabilities via weighted averaging.

    Parameters
    ----------
    proba_dict : Dict[str, np.ndarray]
        Mapping from modality to probability matrix (n_samples, n_classes).
    weights : Optional[Dict[str, float]]
        Weights per modality. If None, equal weights.

    Returns
    -------
    DecisionFusionResult
        Fused probabilities and predictions.
    """
    if not proba_dict:
        raise ValueError("proba_dict must contain at least one modality.")
    modalities = list(proba_dict.keys())
    n_samples = proba_dict[modalities[0]].shape[0]
    n_classes = proba_dict[modalities[0]].shape[1]
    if weights is None:
        weights = {m: 1.0 / len(modalities) for m in modalities}
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        # Normalize
        for m in weights:
            weights[m] /= weight_sum
    for modality in modalities:
        if modality not in weights:
            raise ValueError(f"Modality '{modality}' not in weights.")
        P = proba_dict[modality]
        if P.ndim != 2 or P.shape != (n_samples, n_classes):
            raise ValueError(
                f"Probability matrix for {modality} must be shape "
                f"({n_samples}, {n_classes})."
            )
    P_fused = np.zeros((n_samples, n_classes), dtype=float)
    for modality in modalities:
        P_fused += weights[modality] * proba_dict[modality]
    preds = np.argmax(P_fused, axis=1)
    return DecisionFusionResult(
        predictions=preds, probabilities=P_fused, method="weighted_average"
    )


__all__ = [
    "LateFusionResult",
    "DecisionFusionResult",
    "late_fusion_concat",
    "decision_fusion_vote",
    "decision_fusion_weighted",
]
