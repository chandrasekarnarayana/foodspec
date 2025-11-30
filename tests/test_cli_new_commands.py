from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from foodspec.cli import app
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.data.libraries import create_library

runner = CliRunner()


def _make_hdf5(tmp_path: Path, fname: str) -> Path:
    wavenumbers = np.linspace(1600.0, 1800.0, 50)
    rng = np.random.default_rng(0)
    base = np.exp(-0.5 * ((wavenumbers - 1700) / 15) ** 2)
    X = np.vstack(
        [
            base + rng.normal(0, 0.01, size=wavenumbers.shape),
            base + 0.1 + rng.normal(0, 0.01, size=wavenumbers.shape),
            base * 0.8 + rng.normal(0, 0.01, size=wavenumbers.shape),
            base * 0.9 + rng.normal(0, 0.01, size=wavenumbers.shape),
        ]
    )
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(X))],
            "heating_time": [0, 10, 20, 30],
            "label": ["a", "a", "b", "b"],
        }
    )
    ds = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=metadata, modality="raman")
    path = tmp_path / fname
    create_library(path, ds)
    return path


def _assert_report_files(report_dir: Path, expected: list[str]):
    assert report_dir.exists()
    for name in expected:
        assert (report_dir / name).exists()


def test_cli_heating(tmp_path):
    hdf5_path = _make_hdf5(tmp_path, "heating.h5")
    out_dir = tmp_path / "out_heating"
    result = runner.invoke(
        app, ["heating", str(hdf5_path), "--time-column", "heating_time", "--output-dir", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    reports = list(out_dir.iterdir())
    assert reports
    report_dir = reports[0]
    _assert_report_files(report_dir, ["summary.json", "ratios.csv"])


def test_cli_qc(tmp_path):
    hdf5_path = _make_hdf5(tmp_path, "qc.h5")
    out_dir = tmp_path / "out_qc"
    result = runner.invoke(app, ["qc", str(hdf5_path), "--output-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    reports = list(out_dir.iterdir())
    assert reports
    report_dir = reports[0]
    _assert_report_files(report_dir, ["summary.json", "scores.csv"])


def test_cli_domains(tmp_path):
    hdf5_path = _make_hdf5(tmp_path, "domains.h5")
    out_dir = tmp_path / "out_domains"
    result = runner.invoke(
        app,
        [
            "domains",
            str(hdf5_path),
            "--type",
            "dairy",
            "--label-column",
            "label",
            "--cv-splits",
            "2",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    reports = list(out_dir.iterdir())
    assert reports
    report_dir = reports[0]
    _assert_report_files(report_dir, ["summary.json", "cv_metrics.csv", "confusion_matrix.csv"])
