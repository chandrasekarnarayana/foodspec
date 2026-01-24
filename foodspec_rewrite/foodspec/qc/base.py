"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

QC interfaces and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Protocol

import numpy as np
import pandas as pd


class QCMetric(Protocol):
    """Interface for QC metrics.

    compute must return a DataFrame with one row per sample and metric columns.
    """

    def compute(self, X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass
class QCSummary:
    """Aggregate pass/fail decisions given metric thresholds.

    Parameters
    ----------
    thresholds : Mapping[str, Mapping[str, float]]
        Dict of metric -> bounds. Supported keys per metric: ``min`` and/or ``max``.
    """

    thresholds: Mapping[str, Mapping[str, float]]

    def evaluate(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Return summary with pass flag and fail reasons per sample.

        Raises
        ------
        ValueError
            If any thresholded metric is missing from the provided metrics.
        """

        missing = [m for m in self.thresholds if m not in metrics.columns]
        if missing:
            available = ", ".join(metrics.columns)
            raise ValueError(
                f"Missing metrics for QC thresholds: {', '.join(missing)}. "
                f"Available metrics: {available}."
            )

        summaries: Dict[int, Dict[str, object]] = {}
        for idx, row in metrics.iterrows():
            reasons = []
            for metric, bounds in self.thresholds.items():
                value = row[metric]
                if "min" in bounds and value < bounds["min"]:
                    reasons.append(f"{metric}<{bounds['min']}")
                if "max" in bounds and value > bounds["max"]:
                    reasons.append(f"{metric}>{bounds['max']}")
            summaries[idx] = {
                "pass": len(reasons) == 0,
                "fail_reasons": ";".join(reasons),
            }

        df = pd.DataFrame.from_dict(summaries, orient="index")
        df["pass"] = df["pass"].map(bool).astype(object)
        return df


__all__ = ["QCMetric", "QCSummary"]
