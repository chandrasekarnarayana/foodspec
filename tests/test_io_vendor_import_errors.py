import pathlib

import pytest

from foodspec.io import vendor_formats


def test_read_spc_import_error(tmp_path):
    fake = tmp_path / "fake.spc"
    fake.write_bytes(b"")
    with pytest.raises(ImportError):
        vendor_formats.read_spc(fake)


def test_read_opus_import_error(tmp_path):
    fake = tmp_path / "fake.opus"
    fake.write_bytes(b"")
    with pytest.raises(ImportError):
        vendor_formats.read_opus(fake)
