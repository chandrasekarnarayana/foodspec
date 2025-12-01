import numpy as np
import pandas as pd

from foodspec.apps.meat import run_meat_authentication_workflow
from foodspec.apps.microbial import run_microbial_detection_workflow
from foodspec.core.dataset import FoodSpectrumSet


def _make_ds(labels):
    wn = np.linspace(1000, 1010, 10)
    X = np.random.default_rng(0).normal(size=(len(labels), wn.size))
    meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(len(labels))], "label": labels})
    return FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")


def test_meat_workflow_smoke():
    labels = ["fresh", "fresh", "fresh", "spoiled", "spoiled", "spoiled"]
    ds = _make_ds(labels)
    result = run_meat_authentication_workflow(ds, label_column="label", cv_splits=2)
    assert result.cv_metrics is not None
    assert result.confusion_matrix.shape[0] == len(result.class_labels)


def test_microbial_workflow_smoke():
    labels = ["positive", "positive", "positive", "negative", "negative", "negative"]
    ds = _make_ds(labels)
    result = run_microbial_detection_workflow(ds, label_column="label", cv_splits=2)
    assert result.cv_metrics is not None
    assert result.confusion_matrix.shape[0] == len(result.class_labels)
