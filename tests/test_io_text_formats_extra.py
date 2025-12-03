from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.io.text_formats import read_csv_folder, read_jcamp
from foodspec.io.core import detect_format, read_spectra


def test_read_csv_folder_inconsistent_axes(tmp_path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text("wn,int\n1,10\n2,11\n")
    f2.write_text("wn,int\n1,10\n")  # shorter wn length
    with pytest.raises(ValueError):
        read_csv_folder(tmp_path)


def test_read_csv_folder_no_files(tmp_path):
    with pytest.raises(ValueError):
        read_csv_folder(tmp_path)


def test_read_jcamp_empty(tmp_path):
    f = tmp_path / "empty.jdx"
    f.write_text("##TITLE=empty\n")
    with pytest.raises(ValueError):
        read_jcamp(f)


def test_detect_format_unknown(tmp_path):
    f = tmp_path / "file.unknown"
    f.write_text("data")
    fmt = detect_format(f)
    assert fmt == "unknown"
    with pytest.raises(ValueError):
        read_spectra(f)
