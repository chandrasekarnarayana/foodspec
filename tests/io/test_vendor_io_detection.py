from pathlib import Path

import pytest

from foodspec.spectral_io import detect_format, load_envi, load_opus


def test_detect_format_vendor_mocks():
    base = Path("tests/data/vendor")
    assert detect_format(base / "mock.opus") == "opus"
    assert detect_format(base / "mock.wire.txt") in {"wire", "csv"}
    assert detect_format(base / "mock.hdr") == "envi"


def test_vendor_loader_error_messages():
    base = Path("tests/data/vendor")
    # OPUS stub expects CSV-like; malformed should raise useful error
    with pytest.raises(Exception) as exc:
        load_opus(base / "mock.opus")
    assert "Could not detect" in str(exc).lower() or "failed" in str(exc).lower()
    with pytest.raises(Exception) as exc2:
        load_envi(base / "mock.hdr")
    assert "failed" in str(exc2).lower()
