from pathlib import Path

import pandas as pd

from foodspec.spectral_io import load_opus, load_wire, load_envi, detect_format


def test_vendor_stubs_return_dataset(tmp_path: Path):
    csv = tmp_path / "stub.csv"
    df = pd.DataFrame({"meta": [1, 2], "1000": [1.0, 2.0], "1010": [1.1, 2.1]})
    df.to_csv(csv, index=False)

    ds_opus = load_opus(csv)
    ds_wire = load_wire(csv)
    ds_envi = load_envi(csv)
    for ds in (ds_opus, ds_wire, ds_envi):
        assert ds.spectra.shape[0] == 2
        assert len(ds.wavenumbers) == 2


def test_detect_format():
    assert detect_format("foo.opus") == "opus"
    assert detect_format("foo.wire.txt") in {"wire", "csv"}
    assert detect_format("foo.hdr") == "envi"
    assert detect_format("foo.csv") == "csv"
