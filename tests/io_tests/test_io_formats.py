from pathlib import Path

import numpy as np
import pytest

from foodspec.io.core import detect_format, read_spectra
from foodspec.io.vendor_formats import read_opus, read_spc


def test_detect_format_file_and_folder(tmp_path: Path):
    folder = tmp_path / "csvs"
    folder.mkdir()
    assert detect_format(folder) == "folder_csv"
    f = tmp_path / "file.jdx"
    f.write_text("##TITLE=\n1000 1\n")
    assert detect_format(f) == "jcamp"


def test_read_spectra_csv_and_jcamp():
    fs_csv = read_spectra("tests/data/sample_wide.csv")
    assert fs_csv.x.shape == (2, 3)
    assert fs_csv.wavenumbers[0] == 1000

    fs_jcamp = read_spectra("tests/data/sample_jcamp.jdx")
    assert fs_jcamp.x.shape[0] == 1
    assert np.isclose(fs_jcamp.x[0, 0], 1.0)


def test_vendor_import_errors(tmp_path: Path):
    with pytest.raises(ImportError):
        read_spc(tmp_path / "missing.spc")
    with pytest.raises(ImportError):
        read_opus(tmp_path / "missing.0")
