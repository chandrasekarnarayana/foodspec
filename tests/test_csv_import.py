import numpy as np
import pandas as pd
import pytest

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.io.csv_import import load_csv_spectra


def test_load_csv_wide(tmp_path):
    wn = np.array([1000.0, 1002.0, 1004.0])
    df = pd.DataFrame(
        {
            "wavenumber": wn,
            "s1": [1.0, 1.1, 1.2],
            "s2": [2.0, 2.1, 2.2],
        }
    )
    csv_path = tmp_path / "wide.csv"
    df.to_csv(csv_path, index=False)

    fs = load_csv_spectra(csv_path, format="wide", modality="raman")
    assert isinstance(fs, FoodSpectrumSet)
    assert fs.x.shape == (2, 3)
    assert np.allclose(fs.wavenumbers, wn)
    assert list(fs.metadata["sample_id"]) == ["s1", "s2"]


def test_load_csv_long(tmp_path):
    records = [
        {"sample_id": "a", "wavenumber": 1000.0, "intensity": 1.0, "label": "X"},
        {"sample_id": "a", "wavenumber": 1002.0, "intensity": 1.1, "label": "X"},
        {"sample_id": "a", "wavenumber": 1004.0, "intensity": 1.2, "label": "X"},
        {"sample_id": "b", "wavenumber": 1000.0, "intensity": 2.0, "label": "Y"},
        {"sample_id": "b", "wavenumber": 1002.0, "intensity": 2.1, "label": "Y"},
        {"sample_id": "b", "wavenumber": 1004.0, "intensity": 2.2, "label": "Y"},
    ]
    df = pd.DataFrame.from_records(records)
    csv_path = tmp_path / "long.csv"
    df.to_csv(csv_path, index=False)

    fs = load_csv_spectra(
        csv_path,
        format="long",
        wavenumber_column="wavenumber",
        sample_id_column="sample_id",
        intensity_column="intensity",
        label_column="label",
        modality="ftir",
    )
    assert fs.modality == "ftir"
    assert fs.x.shape == (2, 3)
    assert list(fs.metadata["sample_id"]) == ["a", "b"]
    assert list(fs.metadata["label"]) == ["X", "Y"]


def test_missing_wavenumber_column_raises(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_csv_spectra(csv_path, format="wide", wavenumber_column="wavenumber")


def test_load_csv_long_missing_intensity(tmp_path):
    df = pd.DataFrame({"sample_id": ["a"], "wavenumber": [1000.0], "label": ["x"]})
    csv_path = tmp_path / "bad_long.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_csv_spectra(
            csv_path,
            format="long",
            wavenumber_column="wavenumber",
            sample_id_column="sample_id",
            intensity_column="intensity",
        )


def test_load_csv_invalid_format(tmp_path):
    df = pd.DataFrame({"wavenumber": [1, 2], "s1": [1, 2]})
    csv_path = tmp_path / "fmt.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_csv_spectra(csv_path, format="bad")
