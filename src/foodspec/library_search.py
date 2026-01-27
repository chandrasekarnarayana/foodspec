"""Spectral library search utilities.

Provides similarity metrics (cosine, Pearson, Euclidean, SID, SAM) and
search over a library to return top-k matches with confidence and an
optional overlay plot.

Deprecated shim:
    Use foodspec.workflows.library_search instead. This module will be removed in v2.0.0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine as dist_cosine
from scipy.spatial.distance import euclidean as dist_euclidean
from scipy.stats import pearsonr

warnings.warn(
    "foodspec.library_search is deprecated and will be removed in v2.0.0. "
    "Use foodspec.workflows.library_search instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class MatchResult:
    index: int
    label: str
    score: float
    confidence: float
    metric: str


def _sid(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum() + (q * np.log(q / p)).sum())


def _sam(p: np.ndarray, q: np.ndarray) -> float:
    num = float(np.dot(p, q))
    denom = float(np.linalg.norm(p) * np.linalg.norm(q)) + 1e-12
    angle = np.arccos(np.clip(num / denom, -1.0, 1.0))
    return float(angle)


def compute_similarity(query: np.ndarray, target: np.ndarray, metric: str) -> float:
    metric = metric.lower()
    if metric == "cosine":
        return 1.0 - dist_cosine(query, target)
    if metric == "pearson":
        r, _ = pearsonr(query, target)
        return float(r)
    if metric == "euclidean":
        return -dist_euclidean(query, target)
    if metric == "sid":
        return -_sid(query, target)
    if metric == "sam":
        return -_sam(query, target)
    raise ValueError(f"Unknown metric: {metric}")


def search_library(
    query: np.ndarray,
    library_matrix: np.ndarray,
    labels: List[str] | None = None,
    k: int = 5,
    metric: str = "cosine",
) -> List[MatchResult]:
    scores = [compute_similarity(query, library_matrix[i], metric) for i in range(library_matrix.shape[0])]
    order = np.argsort(scores)[::-1]
    top = order[:k]
    top_scores = np.array(scores)[top]
    # Confidence scaling: normalize to [0,1] by min/max of top-k
    if len(top_scores) > 1:
        s_min, s_max = float(top_scores.min()), float(top_scores.max())
        conf = [(float((s - s_min) / (s_max - s_min + 1e-12))) for s in top_scores]
    else:
        conf = [1.0]
    out: List[MatchResult] = []
    for rank, idx in enumerate(top):
        out.append(
            MatchResult(
                index=int(idx),
                label=str(labels[idx] if labels else idx),
                score=float(scores[idx]),
                confidence=float(conf[rank]),
                metric=metric,
            )
        )
    return out


def overlay_plot(
    query: np.ndarray, wavenumbers: np.ndarray, matches: List[Tuple[str, np.ndarray]], title: str = "Overlay"
):
    plt.figure(figsize=(8, 4))
    plt.plot(wavenumbers, query, label="query", lw=2)
    for label, spectrum in matches:
        plt.plot(wavenumbers, spectrum, label=str(label), alpha=0.7)
    plt.xlabel("Wavenumber")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()
