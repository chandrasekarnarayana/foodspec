import numpy as np
import pytest

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.data.libraries import (
    create_library,
    load_library,
    search_library_fingerprint,
)
from foodspec.data.loader import load_example_oils


def test_load_example_oils():
    ds = load_example_oils()
    assert isinstance(ds, FoodSpectrumSet)
    assert len(ds) > 0
    assert "oil_type" in ds.metadata.columns
    assert ds.x.shape[0] == len(ds.metadata)


def test_library_round_trip(tmp_path):
    ds = load_example_oils()
    path = tmp_path / "lib.h5"
    pytest.importorskip("h5py")
    create_library(path, ds)
    ds_loaded = load_library(path)
    assert np.allclose(ds.x, ds_loaded.x)
    assert np.allclose(ds.wavenumbers, ds_loaded.wavenumbers)
    assert ds_loaded.metadata.equals(ds.metadata)


def test_search_library_fingerprint():
    ds = load_example_oils()
    lib = ds
    # create a query with first two samples identical to library
    query_metadata = ds.metadata.iloc[:2].copy()
    query_metadata["sample_id"] = [f"q{i}" for i in range(2)]
    query = FoodSpectrumSet(
        x=ds.x[:2],
        wavenumbers=ds.wavenumbers,
        metadata=query_metadata,
        modality=ds.modality,
    )
    sims = search_library_fingerprint(lib, query, metric="cosine")
    assert sims.shape == (2, len(lib))
    # Highest similarity should correspond to identical spectra
    top_matches = sims.values.argmax(axis=1)
    assert np.all(top_matches < len(lib))
