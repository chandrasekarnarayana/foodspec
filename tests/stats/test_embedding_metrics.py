import numpy as np

from foodspec.stats.embedding import evaluate_embedding


def test_evaluate_embedding_report_fields():
    # Two clusters in 2D
    X = np.vstack([
        np.random.normal(loc=[0,0], scale=0.1, size=(20,2)),
        np.random.normal(loc=[3,3], scale=0.1, size=(20,2)),
    ])
    y = np.array([0]*20 + [1]*20)
    rep = evaluate_embedding(X, y)
    assert rep.silhouette > 0.5
    assert rep.davies_bouldin < 0.5
    assert rep.between_within_ratio > 1.0
