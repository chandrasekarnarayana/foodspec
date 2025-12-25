import numpy as np
import pandas as pd
import pytest

from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.core.dataset import FoodSpectrumSet

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Permission denied.*joblib.*"),
    pytest.mark.filterwarnings("ignore:.*Passing literal json.*"),
    pytest.mark.filterwarnings("ignore:.*trapz is deprecated.*"),
]


def _synthetic_oil_spectra():
    rng = np.random.default_rng(0)
    wavenumbers = np.linspace(800, 1800, 300)
    n_samples = 40
    labels = np.array(["olive"] * (n_samples // 2) + ["sunflower"] * (n_samples // 2))
    spectra = []
    for i in range(n_samples):
        base = np.zeros_like(wavenumbers)
        # Class-specific peak shifts/intensities
        if labels[i] == "olive":
            base += 1.2 * np.exp(-0.5 * ((wavenumbers - 1655) / 10) ** 2)
            base += 1.0 * np.exp(-0.5 * ((wavenumbers - 1742) / 12) ** 2)
        else:
            base += 0.8 * np.exp(-0.5 * ((wavenumbers - 1655) / 10) ** 2)
            base += 1.3 * np.exp(-0.5 * ((wavenumbers - 1742) / 12) ** 2)
        noise = rng.normal(0, 0.02, size=wavenumbers.shape)
        spectra.append(base + noise)
    X = np.vstack(spectra)
    metadata = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n_samples)], "oil_type": labels})
    return FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=metadata, modality="raman")


def test_run_oil_authentication_workflow():
    ds = _synthetic_oil_spectra()
    result = run_oil_authentication_workflow(ds, label_column="oil_type", classifier_name="rf", cv_splits=3)
    assert not result.cv_metrics.empty
    assert result.confusion_matrix.shape == (2, 2)
    assert set(result.class_labels) == {"olive", "sunflower"}
    # Feature importances available for RF
    if result.feature_importances is not None:
        assert not result.feature_importances.empty
