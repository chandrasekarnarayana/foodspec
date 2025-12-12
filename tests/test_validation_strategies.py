import numpy as np
from sklearn.ensemble import RandomForestClassifier

from foodspec.validation import batch_aware_cv, nested_cv


def test_batch_aware_cv_keeps_batches_together():
    X = np.random.rand(6, 2)
    y = np.array(["A", "A", "B", "B", "C", "C"])
    batches = np.array([1, 1, 2, 2, 3, 3])
    splits = list(batch_aware_cv(X, y, batches, n_splits=3))
    for train_idx, test_idx in splits:
        # ensure no batch is split
        train_batches = set(batches[train_idx])
        test_batches = set(batches[test_idx])
        assert train_batches.isdisjoint(test_batches)


def test_nested_cv_returns_metrics():
    X = np.random.rand(10, 3)
    y = np.array(["A"] * 5 + ["B"] * 5)
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    results = nested_cv(model, X, y, groups=None, outer_splits=2, inner_splits=2)
    assert results
    assert "bal_accuracy" in results[0]
    assert "per_class_recall" in results[0]
