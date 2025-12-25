"""Tests for multimodal dataset functionality."""

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.core.multimodal import MultiModalDataset


def test_multimodal_dataset_construction():
    """Test basic MultiModalDataset construction and validation."""
    wn_raman = np.linspace(800, 1800, 51)
    wn_ftir = np.linspace(400, 4000, 101)
    n_samples = 10

    X_raman = np.random.randn(n_samples, len(wn_raman))
    X_ftir = np.random.randn(n_samples, len(wn_ftir))

    meta = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(n_samples)],
        "label": ["A"] * 5 + ["B"] * 5,
    })

    ds_raman = FoodSpectrumSet(x=X_raman, wavenumbers=wn_raman, metadata=meta.copy(), modality="raman")
    ds_ftir = FoodSpectrumSet(x=X_ftir, wavenumbers=wn_ftir, metadata=meta.copy(), modality="ftir")

    mmd = MultiModalDataset.from_datasets(
        {"raman": ds_raman, "ftir": ds_ftir},
        sample_id_col="sample_id"
    )

    assert len(mmd.modalities()) == 2
    assert "raman" in mmd.modalities()
    assert "ftir" in mmd.modalities()
    assert len(mmd.metadata) == n_samples


def test_multimodal_subset():
    """Test subsetting multimodal dataset."""
    wn = np.linspace(800, 1800, 21)
    n_samples = 6
    X1 = np.random.randn(n_samples, len(wn))
    X2 = np.random.randn(n_samples, len(wn))
    meta = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(n_samples)],
        "label": ["A", "A", "B", "B", "C", "C"],
    })
    ds1 = FoodSpectrumSet(x=X1, wavenumbers=wn, metadata=meta.copy(), modality="raman")
    ds2 = FoodSpectrumSet(x=X2, wavenumbers=wn, metadata=meta.copy(), modality="ftir")
    mmd = MultiModalDataset.from_datasets({"raman": ds1, "ftir": ds2})

    # Filter by label
    mmd_sub = mmd.filter_by_metadata(label="A")
    assert len(mmd_sub.metadata) == 2
    assert all(mmd_sub.metadata["label"] == "A")


def test_multimodal_feature_dict():
    """Test extracting feature dictionary for fusion."""
    wn = np.linspace(800, 1800, 11)
    n = 4
    X1 = np.random.randn(n, len(wn))
    X2 = np.random.randn(n, len(wn))
    meta = pd.DataFrame({"sample_id": [f"S{i}" for i in range(n)]})
    ds1 = FoodSpectrumSet(x=X1, wavenumbers=wn, metadata=meta.copy(), modality="raman")
    ds2 = FoodSpectrumSet(x=X2, wavenumbers=wn, metadata=meta.copy(), modality="nir")
    mmd = MultiModalDataset.from_datasets({"raman": ds1, "nir": ds2})

    feat_dict = mmd.to_feature_dict()
    assert "raman" in feat_dict
    assert "nir" in feat_dict
    assert feat_dict["raman"].shape == (n, len(wn))
    assert feat_dict["nir"].shape == (n, len(wn))
