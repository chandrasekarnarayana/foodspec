from __future__ import annotations

"""Distance-based confidence scoring for library matches.

Maps distances to [0,1] confidence and assigns decisions
('known', 'borderline', 'unknown') using simple thresholding.
"""


from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

__all__ = ["add_confidence", "decision_from_confidence"]


def _scale_distance_to_confidence(distance: float, ref_scale: float) -> float:
    # Inverse scaling: higher distance => lower confidence
    denom = ref_scale + 1e-12
    score = 1.0 / (1.0 + (distance / denom))
    return float(np.clip(score, 0.0, 1.0))


def decision_from_confidence(score: float, thresholds: Dict[str, float] | None = None) -> str:
    t = thresholds or {"known": 0.8, "borderline": 0.5}
    if score >= t["known"]:
        return "known"
    if score >= t["borderline"]:
        return "borderline"
    return "unknown"


def add_confidence(
    sim_table: pd.DataFrame,
    metric: Literal["euclidean", "cosine", "pearson", "sid", "sam"] = "cosine",
    reference_distances: Optional[np.ndarray] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Add confidence and decisions to a similarity table.

    reference_distances: optional array to estimate scale (e.g., median NN distance
    within the library). If not provided, use median distance from the table.
    """
    if "distance" not in sim_table.columns:
        raise ValueError("similarity table must contain a 'distance' column")
    ref_scale = (
        float(np.median(reference_distances))
        if reference_distances is not None
        else float(np.median(sim_table["distance"]))
    )
    scores = [_scale_distance_to_confidence(d, ref_scale) for d in sim_table["distance"].to_numpy()]
    decisions = [decision_from_confidence(s, thresholds) for s in scores]
    out = sim_table.copy()
    out["confidence"] = scores
    out["decision"] = decisions
    return out
