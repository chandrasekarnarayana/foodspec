import pandas as pd
import numpy as np
import pytest

from foodspec.io.base import SpectraLoader
from foodspec.io.core import detect_format, _to_spectrum_set_from_df, read_spectra


def test_spectraloader_protocol_accepts_callable():
    def loader(path: str, **kwargs):
        return pd.DataFrame({"wn": [1, 2], "s1": [0.1, 0.2]})

    fn: SpectraLoader = loader  # type: ignore[assignment]
    df = fn("dummy")
    assert isinstance(df, pd.DataFrame)


def test_detect_format_variants(tmp_path):
    file_csv = tmp_path / "a.csv"; file_csv.write_text("w,n\n1,2")
    file_txt = tmp_path / "b.txt"; file_txt.write_text("w,n\n1,2")
    file_jdx = tmp_path / "c.jdx"; file_jdx.write_text("dummy")
    file_spc = tmp_path / "d.spc"; file_spc.write_text("dummy")
    file_opus = tmp_path / "e.opus"; file_opus.write_text("dummy")

    assert detect_format(file_csv) == "csv"
    assert detect_format(file_txt) == "csv"
    assert detect_format(file_jdx) == "jcamp"
    assert detect_format(file_spc) == "spc"
    assert detect_format(file_opus) == "opus"
    assert detect_format(tmp_path) == "folder_csv"


def test_to_spectrum_set_from_df():
    wn = np.array([1000, 1001, 1002])
    df = pd.DataFrame({"wn": wn, "s1": [1, 2, 3], "s2": [4, 5, 6]})
    fs = _to_spectrum_set_from_df(df)
    assert fs.x.shape == (2, 3)
    assert fs.wavenumbers.tolist() == wn.tolist()
    assert list(fs.metadata["sample_id"]) == ["s1", "s2"]


def test_read_spectra_unknown_raises(tmp_path):
    unknown = tmp_path / "file.xxx"
    unknown.write_text("dummy")
    with pytest.raises(ValueError):
        read_spectra(unknown)
