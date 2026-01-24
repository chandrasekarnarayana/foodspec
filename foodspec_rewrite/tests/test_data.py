"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
"""

import numpy as np
import pandas as pd
import pytest

from foodspec.core.data import Spectrum, SpectraSet


def test_spectrum_validation_and_no_nan() -> None:
    spec = Spectrum(x=[1, 2, 3], y=[4, 5, 6], metadata={"sample_id": "a"})
    assert spec.x.shape == spec.y.shape
    assert spec.metadata["sample_id"] == "a"

    with pytest.raises(ValueError):
        Spectrum(x=[1, 2], y=[1], metadata={})

    with pytest.raises(ValueError):
        Spectrum(x=[1, np.nan], y=[1, 2], metadata={})


def test_spectraset_validation_and_selection() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([100.0, 200.0])
    meta = pd.DataFrame({"sample_id": ["a", "b"], "group": ["g1", "g2"]})
    ds = SpectraSet(X=X, x=x, y=np.array([0, 1]), metadata=meta)

    stats = ds.summary_stats()
    assert np.allclose(stats["mean"], [2.0, 3.0])

    subset = ds.select_by_metadata("group", "g1")
    assert subset.X.shape[0] == 1
    assert subset.metadata.iloc[0]["sample_id"] == "a"


def test_spectraset_export_dataframe() -> None:
    X = np.array([[1.0, 2.0]])
    x = np.array([100.0, 200.0])
    meta = pd.DataFrame({"sample_id": ["a"]})
    ds = SpectraSet(X=X, x=x, y=np.array([1]), metadata=meta)

    df = ds.export_to_dataframe()
    assert "label" in df.columns
    assert "wv_100" in df.columns
    assert df.loc[0, "wv_100"] == 1.0


def test_spectraset_rejects_nan_when_not_allowed() -> None:
    X = np.array([[np.nan, 1.0]])
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        SpectraSet(X=X, x=x, metadata=pd.DataFrame({"sample_id": ["a"]}))
