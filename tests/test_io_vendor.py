from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.io import vendor_formats


def test_read_spc_with_mock(monkeypatch, tmp_path):
    class DummyFile:
        def __init__(self, *_):
            self.x = np.array([1.0, 2.0, 3.0])
            self.y = np.array([10.0, 11.0, 12.0])

    dummy_mod = SimpleNamespace(File=DummyFile)
    monkeypatch.setattr(vendor_formats, "_require", lambda pkgs, extra: dummy_mod)
    fake_file = tmp_path / "sample.spc"
    fake_file.write_bytes(b"mock")
    fs = vendor_formats.read_spc(fake_file)
    assert fs.x.shape == (1, 3)
    assert fs.metadata["sample_id"].iloc[0] == "sample"


def test_read_opus_with_mock(monkeypatch, tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [5, 6, 7]})
    dummy_mod = SimpleNamespace(read_file=lambda path: df)
    monkeypatch.setattr(vendor_formats, "_require", lambda pkgs, extra: dummy_mod)
    fake_file = tmp_path / "sample.0"
    fake_file.write_bytes(b"mock")
    fs = vendor_formats.read_opus(fake_file)
    assert fs.x.shape == (1, 3)
    assert fs.metadata["sample_id"].iloc[0] == "sample"


def test_vendor_missing_dep(monkeypatch, tmp_path):
    def _raise(pkgs, extra):
        raise ImportError("missing")

    monkeypatch.setattr(vendor_formats, "_require", _raise)
    fake_file = tmp_path / "sample.spc"
    fake_file.write_bytes(b"mock")
    with pytest.raises(ImportError):
        vendor_formats.read_spc(fake_file)
