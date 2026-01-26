from __future__ import annotations

import numpy as np

from foodspec.stats.clustering import (
    fuzzy_c_means,
    hierarchical_cluster,
    kmeans_cluster,
    regression_clustering,
)


def test_kmeans_cluster() -> None:
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    res = kmeans_cluster(X, n_clusters=2, random_state=0)
    assert res.labels.shape[0] == 4
    assert res.centers is not None


def test_hierarchical_cluster() -> None:
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    res = hierarchical_cluster(X, n_clusters=2)
    assert res.labels.shape[0] == 4


def test_fuzzy_c_means_memberships() -> None:
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    res = fuzzy_c_means(X, n_clusters=2, random_state=0, max_iter=50)
    row_sums = res.memberships.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_regression_clustering() -> None:
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0.0, 0.1, 1.0, 1.1])
    res = regression_clustering(X, y, n_clusters=2, random_state=0)
    assert len(res.cluster_models) == 2
