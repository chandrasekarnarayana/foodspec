"""Abstention rules and uncertainty-aware interpretability helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from foodspec.trust.metrics import risk_coverage_curve


@dataclass
class AbstentionResult:
    abstain_mask: np.ndarray
    abstain_rate: float
    accuracy_on_answered: Optional[float]
    reasons: Dict[str, int]
    risk_coverage: Dict[str, List[float]]


def _density_threshold(scores: np.ndarray, quantile: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    return float(np.quantile(scores, quantile))


def apply_abstention_rules(
    proba: np.ndarray,
    y_true: Sequence[int],
    *,
    tau: float,
    max_set_size: Optional[int] = None,
    conformal_sets: Optional[Sequence[Iterable[int]]] = None,
    density_scores: Optional[np.ndarray] = None,
    density_quantile: float = 0.05,
) -> AbstentionResult:
    """Apply abstention rules: low confidence, large set size, low density."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if proba.ndim != 2:
        raise ValueError("proba must be 2D")
    if len(y_true) != proba.shape[0]:
        raise ValueError("y_true length must match proba")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    if max_set_size is not None and max_set_size <= 0:
        raise ValueError("max_set_size must be positive")
    if not 0.0 < density_quantile < 1.0:
        raise ValueError("density_quantile must be in (0, 1)")

    max_prob = proba.max(axis=1)
    y_pred = proba.argmax(axis=1)

    reasons: Dict[str, int] = {"low_confidence": 0, "large_set": 0, "low_density": 0}
    abstain_mask = max_prob < tau
    reasons["low_confidence"] = int(np.sum(abstain_mask))

    if conformal_sets is not None:
        if len(conformal_sets) != proba.shape[0]:
            raise ValueError("conformal_sets length must match proba rows")
    if conformal_sets is not None and max_set_size is not None:
        set_sizes = np.array([len(list(s)) for s in conformal_sets], dtype=int)
        large_set = set_sizes > max_set_size
        abstain_mask = abstain_mask | large_set
        reasons["large_set"] = int(np.sum(large_set))

    density_threshold = None
    if density_scores is not None:
        density_scores = np.asarray(density_scores, dtype=float)
        if density_scores.shape[0] != proba.shape[0]:
            raise ValueError("density_scores length must match proba")
        density_threshold = _density_threshold(density_scores, density_quantile)
        low_density = density_scores < density_threshold
        abstain_mask = abstain_mask | low_density
        reasons["low_density"] = int(np.sum(low_density))

    keep = ~abstain_mask
    if keep.any():
        accuracy_on_answered = float(np.mean(y_pred[keep] == y_true[keep]))
    else:
        accuracy_on_answered = None

    abstain_rate = float(np.mean(abstain_mask))
    curve = risk_coverage_curve(y_true, proba)

    return AbstentionResult(
        abstain_mask=abstain_mask,
        abstain_rate=abstain_rate,
        accuracy_on_answered=accuracy_on_answered,
        reasons=reasons,
        risk_coverage=curve,
    )


def filter_importance_by_acceptance(
    importances: np.ndarray,
    accept_mask: np.ndarray,
) -> np.ndarray:
    """Filter or aggregate importances using acceptance mask."""
    importances = np.asarray(importances, dtype=float)
    accept_mask = np.asarray(accept_mask, dtype=bool)
    if importances.shape[0] != accept_mask.shape[0]:
        raise ValueError("importances first dimension must match accept_mask length")
    if importances.ndim == 1:
        return importances[accept_mask]
    return importances[accept_mask, :]


__all__ = ["AbstentionResult", "apply_abstention_rules", "filter_importance_by_acceptance"]
