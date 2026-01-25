"""Abstention utilities for selective classification (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np


@dataclass
class AbstentionResult:
    """Results for abstention rules."""

    predictions: List[int]
    abstain_mask: List[bool]
    accuracy_non_abstained: float | None
    abstain_rate: float
    coverage: float | None = None
    set_sizes: List[int] | None = None
    confidence_scores: Optional[np.ndarray] = None
    reason_codes: List[str] = field(default_factory=list)

    @property
    def abstention_rate(self) -> float:
        """Alias for abstain_rate for compatibility."""
        return self.abstain_rate

    @property
    def accuracy_on_answered(self) -> float | None:
        """Alias for accuracy_non_abstained for consistency."""
        return self.accuracy_non_abstained

    @property
    def coverage_on_answered(self) -> float | None:
        """Coverage metric (alias)."""
        return self.coverage


def evaluate_abstention(
    proba: np.ndarray,
    y_true: Sequence[int],
    threshold: float,
    prediction_sets: Sequence[Iterable[int]] | None = None,
    max_set_size: int | None = None,
) -> AbstentionResult:
    """Apply abstention rules and summarize performance."""
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y_true)
    if p.ndim != 2:
        raise ValueError("proba must be 2D (n_samples, n_classes)")
    if p.shape[0] != y.shape[0]:
        raise ValueError("proba and y_true must have the same length")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    max_probs = p.max(axis=1)
    preds = p.argmax(axis=1)
    preds = np.asarray(preds)
    y = np.asarray(y)
    abstain_low_conf = max_probs <= threshold

    set_sizes: List[int] | None = None
    abstain_large_set = np.zeros_like(abstain_low_conf, dtype=bool)
    if prediction_sets is not None:
        if max_set_size is None:
            raise ValueError("max_set_size must be provided when prediction_sets are given")
        if max_set_size <= 0:
            raise ValueError("max_set_size must be positive")
        if len(prediction_sets) != p.shape[0]:
            raise ValueError("prediction_sets length must match proba rows")
        set_sizes = [len(list(s)) for s in prediction_sets]
        abstain_large_set = np.array([sz > max_set_size for sz in set_sizes], dtype=bool)

    abstain_mask = (abstain_low_conf | abstain_large_set).astype(bool)
    keep_mask = np.asarray(~abstain_mask)

    accuracy_non_abstained: float | None
    if keep_mask.any():
        accuracy_non_abstained = float(np.mean(preds[keep_mask] == y[keep_mask]))
    else:
        accuracy_non_abstained = None

    abstain_rate = float(np.mean(abstain_mask))

    coverage: float | None = None
    if prediction_sets is not None:
        contains_true = [int(y[i] in prediction_sets[i]) for i in range(p.shape[0])]
        coverage = float(np.mean(contains_true))

    return AbstentionResult(
        predictions=preds.astype(int).tolist(),
        abstain_mask=abstain_mask.tolist(),
        accuracy_non_abstained=accuracy_non_abstained,
        abstain_rate=abstain_rate,
        coverage=coverage,
        set_sizes=set_sizes,
    )


class MaxProbAbstainer:
    """Abstain when max probability is below threshold."""

    def __init__(self, threshold: float):
        """Initialize MaxProbAbstainer."""
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold

    def apply(
        self, proba: np.ndarray, conformal_sets: Sequence[Iterable[int]] | None = None
    ) -> tuple[np.ndarray, List[str]]:
        """Apply abstention rule."""
        p = np.asarray(proba, dtype=float)
        if p.ndim != 2:
            raise ValueError(f"proba must be 2D, got shape {p.shape}")

        max_probs = p.max(axis=1)
        mask = max_probs <= self.threshold

        reasons = ["low_confidence" if m else "confident" for m in mask]
        return mask, reasons

    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        mask_abstain: Sequence[bool],
    ) -> Dict[str, float | None]:
        """Evaluate abstention performance."""
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        mask = np.asarray(mask_abstain, dtype=bool)

        if len(y_t) != len(y_p) or len(y_t) != len(mask):
            raise ValueError("y_true, y_pred, mask_abstain must have same length")

        abstain_rate = float(np.mean(mask))
        keep_mask = ~mask

        if keep_mask.any():
            accuracy_on_answered = float(np.mean(y_p[keep_mask] == y_t[keep_mask]))
        else:
            accuracy_on_answered = None

        return {
            "abstain_rate": abstain_rate,
            "accuracy_on_answered": accuracy_on_answered,
            "coverage_on_answered": None,
        }


class ConformalSizeAbstainer:
    """Abstain when conformal prediction set size exceeds maximum."""

    def __init__(self, max_set_size: int):
        """Initialize ConformalSizeAbstainer."""
        if max_set_size <= 0:
            raise ValueError(f"max_set_size must be positive, got {max_set_size}")
        self.max_set_size = max_set_size

    def apply(
        self, proba: np.ndarray | None, conformal_sets: Sequence[Iterable[int]]
    ) -> tuple[np.ndarray, List[str]]:
        """Apply abstention rule."""
        if conformal_sets is None:
            raise ValueError("conformal_sets cannot be None for ConformalSizeAbstainer")

        set_sizes = [len(list(s)) for s in conformal_sets]
        mask = np.array([sz > self.max_set_size for sz in set_sizes], dtype=bool)
        reasons = ["large_set" if m else "small_set" for m in mask]
        return mask, reasons

    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        mask_abstain: Sequence[bool],
    ) -> Dict[str, float | None]:
        """Evaluate abstention performance."""
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        mask = np.asarray(mask_abstain, dtype=bool)

        if len(y_t) != len(y_p) or len(y_t) != len(mask):
            raise ValueError("y_true, y_pred, mask_abstain must have same length")

        abstain_rate = float(np.mean(mask))
        keep_mask = ~mask

        if keep_mask.any():
            accuracy_on_answered = float(np.mean(y_p[keep_mask] == y_t[keep_mask]))
        else:
            accuracy_on_answered = None

        return {
            "abstain_rate": abstain_rate,
            "accuracy_on_answered": accuracy_on_answered,
            "coverage_on_answered": None,
        }


class CombinedAbstainer:
    """Combine multiple abstention policies with logical operators."""

    def __init__(
        self,
        rules: List[MaxProbAbstainer | ConformalSizeAbstainer],
        mode: Literal["any", "all"] = "any",
    ):
        """Initialize CombinedAbstainer."""
        if not rules:
            raise ValueError("rules cannot be empty")
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got {mode}")
        self.rules = rules
        self.mode = mode

    def apply(
        self, proba: np.ndarray | None, conformal_sets: Sequence[Iterable[int]] | None = None
    ) -> tuple[np.ndarray, List[str]]:
        """Apply combined abstention rule."""
        masks_list = []
        reasons_list = []

        for rule in self.rules:
            if isinstance(rule, MaxProbAbstainer):
                mask, reasons = rule.apply(proba, conformal_sets)
            elif isinstance(rule, ConformalSizeAbstainer):
                mask, reasons = rule.apply(proba, conformal_sets)
            else:
                raise ValueError(f"Unknown rule type: {type(rule)}")

            masks_list.append(mask)
            reasons_list.append(reasons)

        # Combine masks
        masks_array = np.array(masks_list, dtype=bool)
        if self.mode == "any":
            combined_mask = np.any(masks_array, axis=0)
        else:  # all
            combined_mask = np.all(masks_array, axis=0)

        # Combine reasons per sample
        combined_reasons = []
        for i in range(masks_array.shape[1]):
            sample_reasons = [reasons_list[j][i] for j in range(len(self.rules))]
            combined_reasons.append("|".join(sample_reasons))

        return combined_mask, combined_reasons

    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        mask_abstain: Sequence[bool],
    ) -> Dict[str, float | None]:
        """Evaluate combined abstention performance."""
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        mask = np.asarray(mask_abstain, dtype=bool)

        if len(y_t) != len(y_p) or len(y_t) != len(mask):
            raise ValueError("y_true, y_pred, mask_abstain must have same length")

        abstain_rate = float(np.mean(mask))
        keep_mask = ~mask

        if keep_mask.any():
            accuracy_on_answered = float(np.mean(y_p[keep_mask] == y_t[keep_mask]))
        else:
            accuracy_on_answered = None

        return {
            "abstain_rate": abstain_rate,
            "accuracy_on_answered": accuracy_on_answered,
            "coverage_on_answered": None,
        }


__all__ = [
    "AbstentionResult",
    "evaluate_abstention",
    "MaxProbAbstainer",
    "ConformalSizeAbstainer",
    "CombinedAbstainer",
]
