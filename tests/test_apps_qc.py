import numpy as np
import pandas as pd

from foodspec.apps.qc import apply_qc_model, train_qc_model
from foodspec.core.dataset import FoodSpectrumSet


def test_qc_oneclass_svm_labels_inliers_outliers():
    rng = np.random.default_rng(0)
    # Inliers cluster
    inliers = rng.normal(loc=0.0, scale=0.2, size=(20, 5))
    # Outliers far away
    outliers = rng.normal(loc=3.0, scale=0.1, size=(4, 5))
    X = np.vstack([inliers, outliers])
    wn = np.linspace(800, 1000, X.shape[1])
    metadata = pd.DataFrame({"sample_id": [f"s{i}" for i in range(len(X))]})
    ds = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=metadata, modality="raman")

    train_mask = pd.Series([True] * len(inliers) + [False] * len(outliers))
    model = train_qc_model(ds, train_mask=train_mask, model_type="oneclass_svm", gamma="scale", nu=0.1)
    result = apply_qc_model(ds, model=model)

    assert np.isfinite(result.scores.to_numpy()).all()
    labels = result.labels_pred.to_numpy()
    assert (labels[: len(inliers)] == "authentic").sum() >= len(inliers) - 2
    assert (labels[len(inliers) :] == "suspect").sum() >= len(outliers) - 1

