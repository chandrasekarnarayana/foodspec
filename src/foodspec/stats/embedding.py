from __future__ import annotations
"""
Embedding metrics and representation evaluation.
"""


from dataclasses import dataclass

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


@dataclass
class EmbeddingReport:
    silhouette: float
    davies_bouldin: float
    between_within_ratio: float
    stability: float


def evaluate_embedding(X_emb: np.ndarray, labels: np.ndarray) -> EmbeddingReport:
    """Evaluate embedding quality using multiple metrics.

    Args:
        X_emb: Embedded data (n_samples, n_dimensions).
        labels: Class labels for each sample.

    Returns:
        An `EmbeddingReport` with silhouette score, Davies-Bouldin index,
        between/within variance ratio, and stability (bootstrap mean silhouette).

    Examples:
        >>> from foodspec.stats.embedding import evaluate_embedding
        >>> import numpy as np
        >>> X_emb = np.random.randn(100, 2)
        >>> labels = np.repeat([0, 1, 2], [30, 40, 30])
        >>> report = evaluate_embedding(X_emb, labels)
        >>> hasattr(report, 'silhouette')
        True
    """
    sil = float(silhouette_score(X_emb, labels))
    db = float(davies_bouldin_score(X_emb, labels))
    # Between/within using simple variance decomposition
    grand = X_emb.mean(axis=0)
    classes = np.unique(labels)
    between = 0.0
    within = 0.0
    for c in classes:
        Xc = X_emb[labels == c]
        mu = Xc.mean(axis=0)
        between += float(len(Xc) * np.sum((mu - grand) ** 2))
        within += float(np.sum((Xc - mu) ** 2))
    ratio = float(between / (within + 1e-12))
    # Bootstrap stability: average silhouette over bootstrap resamples
    rng = np.random.default_rng(0)
    n = X_emb.shape[0]
    sil_boot = []
    for _ in range(50):
        idx = rng.integers(0, n, size=n)
        sil_boot.append(float(silhouette_score(X_emb[idx], labels[idx])))
    stability = float(np.mean(sil_boot))
    return EmbeddingReport(silhouette=sil, davies_bouldin=db, between_within_ratio=ratio, stability=stability)
