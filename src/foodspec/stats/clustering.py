"""
Clustering utilities for exploratory analysis and QC diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

try:  # optional
    from scipy.cluster.hierarchy import linkage
except Exception:  # pragma: no cover
    linkage = None


@dataclass
class ClusteringResult:
    labels: np.ndarray
    centers: Optional[np.ndarray]
    inertia: Optional[float]
    silhouette: Optional[float]


def kmeans_cluster(X: np.ndarray, n_clusters: int, *, random_state: int = 0, n_init: int = 10) -> ClusteringResult:
    """Run K-means clustering and return labels and diagnostics."""
    X = np.asarray(X, dtype=float)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    sil = None
    if n_clusters > 1 and X.shape[0] > n_clusters:
        sil = float(silhouette_score(X, labels))
    return ClusteringResult(
        labels=labels,
        centers=model.cluster_centers_,
        inertia=float(model.inertia_),
        silhouette=sil,
    )


@dataclass
class HierarchicalClusteringResult:
    labels: np.ndarray
    linkage_matrix: Optional[np.ndarray]
    silhouette: Optional[float]


def hierarchical_cluster(
    X: np.ndarray,
    n_clusters: int,
    *,
    method: str = "ward",
) -> HierarchicalClusteringResult:
    """Run hierarchical clustering with optional linkage matrix output."""
    X = np.asarray(X, dtype=float)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(X)
    sil = None
    if n_clusters > 1 and X.shape[0] > n_clusters:
        sil = float(silhouette_score(X, labels))
    link = linkage(X, method=method) if linkage is not None else None
    return HierarchicalClusteringResult(labels=labels, linkage_matrix=link, silhouette=sil)


@dataclass
class FuzzyCMeansResult:
    memberships: np.ndarray
    centers: np.ndarray
    objective: float


def fuzzy_c_means(
    X: np.ndarray,
    n_clusters: int,
    *,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-5,
    random_state: int = 0,
) -> FuzzyCMeansResult:
    """Fuzzy C-means clustering (no external dependencies)."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    memberships = rng.random((n_samples, n_clusters))
    memberships = memberships / memberships.sum(axis=1, keepdims=True)
    centers = np.zeros((n_clusters, X.shape[1]))
    obj = 0.0

    for _ in range(max_iter):
        weights = memberships**m
        centers = (weights.T @ X) / (weights.sum(axis=0)[:, None] + 1e-12)
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2) + 1e-12
        inv = distances ** (-2 / (m - 1))
        new_memberships = inv / inv.sum(axis=1, keepdims=True)
        obj_new = float(np.sum((weights) * (distances**2)))
        if abs(obj_new - obj) < tol:
            obj = obj_new
            memberships = new_memberships
            break
        obj = obj_new
        memberships = new_memberships

    return FuzzyCMeansResult(memberships=memberships, centers=centers, objective=obj)


@dataclass
class RegressionClusterResult:
    labels: np.ndarray
    cluster_models: Dict[int, Dict[str, float]]


def regression_clustering(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
) -> RegressionClusterResult:
    """Cluster samples then fit per-cluster linear regressions."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    cluster_models: Dict[int, Dict[str, float]] = {}

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if mask.sum() < 2:
            cluster_models[cluster_id] = {"slope": 0.0, "intercept": float(np.mean(y[mask])) if mask.any() else 0.0}
            continue
        reg = LinearRegression().fit(X[mask], y[mask])
        r2 = reg.score(X[mask], y[mask])
        cluster_models[cluster_id] = {
            "slope": float(reg.coef_.ravel()[0]) if reg.coef_.size else 0.0,
            "intercept": float(reg.intercept_.ravel()[0])
            if hasattr(reg.intercept_, "ravel")
            else float(reg.intercept_),
            "r2": float(r2),
        }
    return RegressionClusterResult(labels=labels, cluster_models=cluster_models)


__all__ = [
    "ClusteringResult",
    "HierarchicalClusteringResult",
    "FuzzyCMeansResult",
    "RegressionClusterResult",
    "kmeans_cluster",
    "hierarchical_cluster",
    "fuzzy_c_means",
    "regression_clustering",
]
