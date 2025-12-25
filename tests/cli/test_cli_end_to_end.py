import numpy as np
import pandas as pd
from typer.testing import CliRunner

from foodspec.cli import app


def _write_spectrum(path, wavenumbers, intensities):
    data = np.column_stack([wavenumbers, intensities])
    np.savetxt(path, data)


def test_cli_end_to_end(tmp_path):
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    wavenumbers = np.linspace(800.0, 1200.0, 21)
    labels = ["OilA"] * 5 + ["OilB"] * 5
    files = []
    for i, lbl in enumerate(labels):
        fname = f"s{i + 1}.txt"
        path = raw_dir / fname
        if lbl == "OilA":
            intensities = 1.0 + np.exp(-0.5 * ((wavenumbers - 1000) / 20) ** 2)
        else:
            intensities = 1.0 + np.exp(-0.5 * ((wavenumbers - 1100) / 20) ** 2)
        _write_spectrum(path, wavenumbers, intensities)
        files.append(fname)

    meta = pd.DataFrame(
        {
            "sample_id": [f.split(".")[0] for f in files],
            "oil_type": labels,
        }
    )
    meta_path = tmp_path / "metadata.csv"
    meta.to_csv(meta_path, index=False)

    hdf5_path = tmp_path / "preprocessed.h5"
    result_pre = runner.invoke(
        app,
        [
            "preprocess",
            str(raw_dir),
            str(hdf5_path),
            "--metadata-csv",
            str(meta_path),
            "--modality",
            "raman",
            "--min-wn",
            "600",
            "--max-wn",
            "1800",
        ],
    )
    assert result_pre.exit_code == 0, result_pre.output
    assert hdf5_path.exists()

    report_path = tmp_path / "oil_auth_report.html"
    result_auth = runner.invoke(
        app,
        [
            "oil-auth",
            str(hdf5_path),
            "--label-column",
            "oil_type",
            "--output-report",
            str(report_path),
        ],
    )
    assert result_auth.exit_code == 0, result_auth.output
    assert report_path.exists()
    assert report_path.stat().st_size > 0
