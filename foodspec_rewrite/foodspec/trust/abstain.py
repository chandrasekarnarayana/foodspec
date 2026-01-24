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

Abstention utilities for classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class AbstentionResult:
    """Results for abstention rules.

    Attributes
    ----------
    predictions : list of int
        Argmax predictions per sample.
    abstain_mask : list of bool
        True if sample was abstained.
    accuracy_non_abstained : float | None
        Accuracy on non-abstained samples; None if all abstained.
    abstain_rate : float
        Fraction of samples abstained.
    coverage : float | None
        Fraction of samples whose true label is inside the provided prediction set;
        None if prediction_sets were not provided.
    set_sizes : list of int | None
        Sizes of provided prediction sets (or None if not provided).
    """

    predictions: List[int]
    abstain_mask: List[bool]
    accuracy_non_abstained: float | None
    abstain_rate: float
    coverage: float | None
    set_sizes: List[int] | None


def evaluate_abstention(
    proba: np.ndarray,
    y_true: Sequence[int],
    threshold: float,
    prediction_sets: Sequence[Iterable[int]] | None = None,
    max_set_size: int | None = None,
) -> AbstentionResult:
    """Apply abstention rules and summarize performance.

    Rules: abstain if ``max(proba) < threshold`` OR (when provided)
    ``len(prediction_set) > max_set_size``.

    Parameters
    ----------
    proba : ndarray, shape (n_samples, n_classes)
        Predicted class probabilities.
    y_true : sequence of int
        Ground truth labels.
    threshold : float
        Confidence threshold below which to abstain.
    prediction_sets : sequence of iterables of int, optional
        Conformal prediction sets aligned with samples.
    max_set_size : int, optional
        Maximum allowed prediction set size; samples exceeding this abstain.

    Returns
    -------
    AbstentionResult

    Examples
    --------
    >>> import numpy as np
    >>> proba = np.array([[0.7, 0.3], [0.55, 0.45]])
    >>> y_true = [0, 1]
    >>> res = evaluate_abstention(proba, y_true, threshold=0.6)
    >>> res.abstain_mask
    [False, True]
    >>> res.accuracy_non_abstained
    1.0
    """

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
    abstain_low_conf = max_probs < threshold

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
    keep_mask = ~abstain_mask

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


__all__ = ["AbstentionResult", "evaluate_abstention"]
