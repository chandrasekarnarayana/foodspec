"""
Abstention utilities for selective classification.

Provides principled rejection rules based on confidence thresholds,
prediction set sizes, and coverage constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class AbstentionResult:
    """Result of abstention evaluation."""
    
    abstain_mask: np.ndarray
    """Boolean mask indicating abstained samples."""
    
    predictions: np.ndarray
    """Predicted labels (NaN for abstained)."""
    
    abstention_rate: float
    """Fraction of samples abstained."""
    
    accuracy_non_abstained: Optional[float] = None
    """Accuracy on samples where prediction made (higher = better)."""
    
    accuracy_abstained: Optional[float] = None
    """Accuracy on samples where abstained (for analysis only)."""
    
    coverage: Optional[float] = None
    """Coverage on non-abstained samples (P(y ∈ pred | not abstained))."""
    
    confidence_scores: Optional[np.ndarray] = None
    """Max predicted probability for each sample."""

    @property
    def abstain_rate(self) -> float:
        """Backward-compatible alias for abstention_rate."""
        return self.abstention_rate


def evaluate_abstention(
    proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.7,
    prediction_sets: Optional[List[List[int]]] = None,
    max_set_size: Optional[int] = None,
) -> AbstentionResult:
    """
    Evaluate abstention rules on predictions.
    
    Abstains (rejects) when:
    1. max(predicted prob) < threshold, OR
    2. |prediction_set| > max_set_size (if provided)
    
    Args:
        proba: Predicted probabilities shape (n, n_classes)
        y_true: True labels shape (n,)
        threshold: Confidence threshold for abstention (0, 1)
        prediction_sets: Optional list of prediction sets (for set size rule)
        max_set_size: Optional max set size threshold
    
    Returns:
        AbstentionResult with decisions, accuracies, and coverage
    
    Raises:
        ValueError: If threshold not in (0, 1) or shapes don't match
    """
    if not (0 < threshold < 1):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    
    if len(proba) != len(y_true):
        raise ValueError(
            f"proba and y_true shape mismatch: {len(proba)} vs {len(y_true)}"
        )
    
    y_true = np.asarray(y_true)
    n_samples = len(proba)
    
    # Get confidence scores
    confidence = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    
    # Abstention by confidence
    # Abstain when confidence does not exceed the threshold (strict > rule)
    abstain_conf = confidence <= threshold
    
    # Abstention by prediction set size
    abstain_size = np.zeros(n_samples, dtype=bool)
    if prediction_sets is not None and max_set_size is not None:
        abstain_size = np.array([
            len(pred_set) > max_set_size
            for pred_set in prediction_sets
        ])
    
    # Combined abstention mask
    abstain_mask = abstain_conf | abstain_size
    
    # Accuracy on non-abstained
    accuracy_non_abstained = None
    accuracy_abstained = None
    coverage = None
    
    if np.any(~abstain_mask):
        # Accuracy on predictions made
        mask_pred = ~abstain_mask
        accuracy_non_abstained = np.mean(predictions[mask_pred] == y_true[mask_pred])
        
        # Coverage on prediction sets (if provided)
        if prediction_sets is not None:
            covered = np.array([
                y_true[i] in prediction_sets[i]
                for i in range(n_samples)
            ])
            coverage = np.mean(covered[mask_pred])
    
    if np.any(abstain_mask):
        # Accuracy on abstained (informative but not used in decision)
        mask_abstain = abstain_mask
        accuracy_abstained = np.mean(predictions[mask_abstain] == y_true[mask_abstain])
    
    abstention_rate = np.mean(abstain_mask)
    
    # Create predictions array with NaN for abstained
    predictions_out = predictions.astype(float)
    predictions_out[abstain_mask] = np.nan
    
    return AbstentionResult(
        abstain_mask=abstain_mask,
        predictions=predictions_out,
        abstention_rate=abstention_rate,
        accuracy_non_abstained=accuracy_non_abstained,
        accuracy_abstained=accuracy_abstained,
        coverage=coverage,
        confidence_scores=confidence,
    )


__all__ = ["evaluate_abstention", "AbstentionResult"]
