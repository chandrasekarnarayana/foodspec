"""Confidence guards for predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ConfidenceWarning:
    level: str  # info|warn|critical
    reason: str
    recommendation: str


def _ensure_probabilities(probs: np.ndarray | float | list[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 0:
        arr = np.array([arr, 1 - arr])
    if arr.ndim == 1 and arr.size == 2:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = np.hstack([1 - arr, arr])
    return arr


def guard_prediction(
    probs: np.ndarray | float | list[float],
    min_confidence: float = 0.6,
    entropy_threshold: float = 0.8,
    margin_threshold: float = 0.2,
) -> Tuple[bool, List[ConfidenceWarning]]:
    """Assess whether a prediction is trustworthy.

    Returns (do_not_trust, warnings).
    """
    p = _ensure_probabilities(probs)
    if p.ndim == 2:
        # Take max probability per row for batch use
        max_probs = p.max(axis=1)
        entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
        margins = np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]
        max_p = float(max_probs.mean())
        ent = float(entropy.mean())
        margin = float(margins.mean())
    else:
        max_p = float(p.max())
        entropy = -np.sum(p * np.log(p + 1e-12))
        ent = float(entropy)
        top_two = np.sort(p)
        margin = float(top_two[-1] - top_two[-2]) if top_two.size >= 2 else 1.0

    warnings: List[ConfidenceWarning] = []

    if max_p < min_confidence:
        warnings.append(
            ConfidenceWarning(
                level="critical",
                reason=f"Low maximum confidence ({max_p:.2f} < {min_confidence:.2f}).",
                recommendation="Hold prediction or route to human review.",
            )
        )

    if ent > entropy_threshold:
        warnings.append(
            ConfidenceWarning(
                level="warn",
                reason=f"High entropy ({ent:.2f} > {entropy_threshold:.2f}).",
                recommendation="Consider calibration or ensembling.",
            )
        )

    if margin < margin_threshold:
        warnings.append(
            ConfidenceWarning(
                level="warn",
                reason=f"Small class margin ({margin:.2f} < {margin_threshold:.2f}).",
                recommendation="Check class overlap or adjust decision threshold.",
            )
        )

    do_not_trust = any(w.level == "critical" for w in warnings)
    return do_not_trust, warnings
