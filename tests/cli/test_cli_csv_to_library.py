import subprocess
import sys
from pathlib import Path

from foodspec.io import load_library


def _write_wide_csv(path: Path) -> None:
    path.write_text(
        """wavenumber,sample_001,sample_002\n1000,1.0,1.5\n1010,2.0,2.5\n1020,3.0,3.5\n""",
        encoding="utf-8",
    )


def _write_long_csv(path: Path) -> None:
    path.write_text(
        """sample_id,wavenumber,intensity,oil_type\ns1,1000,1.0,VO\ns1,1010,2.0,VO\ns1,1020,3.0,VO\ns2,1000,1.5,PO\ns2,1010,2.5,PO\ns2,1020,3.5,PO\n""",
        encoding="utf-8",
    )


def test_cli_csv_to_library_wide(tmp_path: Path):
    csv = tmp_path / "wide.csv"
    _write_wide_csv(csv)
    out_h5 = tmp_path / "wide.h5"
    cmd = [
        sys.executable,
        "-m",
        "foodspec.cli",
        "csv-to-library",
        str(csv),
        str(out_h5),
        "--format",
        "wide",
        "--modality",
        "raman",
        "--wavenumber-column",
        "wavenumber",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    ds = load_library(str(out_h5))
    assert len(ds) == 2
    assert ds.modality == "raman"
    assert ds.wavenumbers.shape[0] == 3


def test_cli_csv_to_library_long(tmp_path: Path):
    csv = tmp_path / "long.csv"
    _write_long_csv(csv)
    out_h5 = tmp_path / "long.h5"
    cmd = [
        sys.executable,
        "-m",
        "foodspec.cli",
        "csv-to-library",
        str(csv),
        str(out_h5),
        "--format",
        "long",
        "--modality",
        "raman",
        "--wavenumber-column",
        "wavenumber",
        "--sample-id-column",
        "sample_id",
        "--intensity-column",
        "intensity",
        "--label-column",
        "oil_type",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    ds = load_library(str(out_h5))
    assert len(ds) == 2
    assert ds.modality == "raman"
    assert ds.wavenumbers.shape[0] == 3
    # labels from long CSV copied to metadata
    assert "oil_type" in ds.metadata.columns
