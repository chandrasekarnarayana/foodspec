"""Prediction quality gating for production use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from foodspec.predict.guards import guard_prediction


@dataclass
class PredictionQCResult:
    do_not_trust: bool
    warnings: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.do_not_trust:
            return "🚫 Do not trust prediction: " + "; ".join(self.reasons)
        return "✅ Prediction passed QC." + (" " + "; ".join(self.warnings) if self.warnings else "")


def evaluate_prediction_qc(
    probs: np.ndarray | float | list[float],
    drift_score: Optional[float] = None,
    ece: Optional[float] = None,
    min_confidence: float = 0.6,
    drift_threshold: float = 0.2,
    ece_threshold: float = 0.08,
) -> PredictionQCResult:
    """Aggregate confidence, drift, and calibration checks."""

    do_not_trust, guard_warnings = guard_prediction(probs, min_confidence=min_confidence)
    warnings = [w.reason for w in guard_warnings]
    reasons: List[str] = []

    if drift_score is not None and drift_score > drift_threshold:
        reasons.append(f"Production drift high ({drift_score:.2f} > {drift_threshold:.2f}).")
        do_not_trust = True

    if ece is not None and ece > ece_threshold:
        reasons.append(f"Calibration error elevated (ECE {ece:.2f} > {ece_threshold:.2f}).")
        do_not_trust = True

    return PredictionQCResult(do_not_trust=do_not_trust, warnings=warnings, reasons=reasons)
