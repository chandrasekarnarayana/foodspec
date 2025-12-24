from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from foodspec.cli import app
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.data.libraries import create_library

runner = CliRunner()


def _make_hdf5(tmp_path: Path, fname: str, with_labels: bool = True) -> Path:
    """Create a small synthetic HDF5 library."""

    wn = np.linspace(1600, 1800, 20)
    rng = np.random.default_rng(0)
    base = np.exp(-0.5 * ((wn - 1700) / 15) ** 2)
    X = np.vstack(
        [
            base + rng.normal(0, 0.01, size=wn.shape),
            base * 1.1 + rng.normal(0, 0.01, size=wn.shape),
            base * 0.9 + rng.normal(0, 0.01, size=wn.shape),
            base * 1.05 + rng.normal(0, 0.01, size=wn.shape),
        ]
    )
    meta = {"sample_id": [f"s{i}" for i in range(X.shape[0])]}
    if with_labels:
        meta["oil_type"] = ["A", "A", "B", "B"]
        meta["label"] = ["a", "a", "b", "b"]
    meta["heating_time"] = [0, 10, 20, 30]
    ds = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=pd.DataFrame(meta), modality="raman")
    path = tmp_path / fname
    create_library(path, ds)
    return path


def _assert_report_created(out_dir: Path):
    reports = [p for p in out_dir.iterdir() if p.is_dir()]
    assert reports, f"No report folder found in {out_dir}"
    found = any((rep / "summary.json").exists() for rep in reports)
    assert found


def test_cli_preprocess(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    wn = np.linspace(600, 800, 10)
    for i in range(3):
        data = np.column_stack([wn, np.sin(wn / 50) + i])
        (raw / f"spec_{i}.txt").write_text("\n".join(" ".join(map(str, row)) for row in data))
    out_h5 = tmp_path / "out.h5"
    result = runner.invoke(
        app,
        [
            "preprocess",
            str(raw),
            str(out_h5),
            "--min-wn",
            "600",
            "--max-wn",
            "800",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_h5.exists()


def test_cli_oil_auth(tmp_path: Path):
    hdf5_path = _make_hdf5(tmp_path, "oil.h5", with_labels=True)
    out_dir = tmp_path / "oil_reports"
    report_html = out_dir / "report.html"
    result = runner.invoke(
        app,
        [
            "oil-auth",
            str(hdf5_path),
            "--label-column",
            "oil_type",
            "--cv-splits",
            "2",
            "--save-model",
            str(out_dir / "model"),
            "--output-report",
            str(report_html),
        ],
    )
    assert result.exit_code == 0, result.output
    _assert_report_created(out_dir)


def test_cli_heating(tmp_path: Path):
    hdf5_path = _make_hdf5(tmp_path, "heating.h5", with_labels=True)
    out_dir = tmp_path / "heating_reports"
    result = runner.invoke(
        app,
        ["heating", str(hdf5_path), "--time-column", "heating_time", "--output-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    _assert_report_created(out_dir)


def test_cli_qc(tmp_path: Path):
    hdf5_path = _make_hdf5(tmp_path, "qc.h5", with_labels=False)
    out_dir = tmp_path / "qc_reports"
    result = runner.invoke(app, ["qc", str(hdf5_path), "--output-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    _assert_report_created(out_dir)


def test_cli_domains(tmp_path: Path):
    hdf5_path = _make_hdf5(tmp_path, "domains.h5", with_labels=True)
    out_dir = tmp_path / "domain_reports"
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
    _assert_report_created(out_dir)


def test_cli_mixture(tmp_path: Path):
    wn = np.linspace(1600, 1650, 20)
    comp1 = np.exp(-0.5 * ((wn - 1610) / 2) ** 2)
    comp2 = 0.5 * np.exp(-0.5 * ((wn - 1635) / 3) ** 2)
    pure_ds = FoodSpectrumSet(
        x=np.vstack([comp1, comp2]),
        wavenumbers=wn,
        metadata=pd.DataFrame({"sample_id": ["c1", "c2"]}),
        modality="raman",
    )
    pure_path = tmp_path / "pure.h5"
    create_library(pure_path, pure_ds)
    mix_spec = 0.6 * comp1 + 0.4 * comp2
    mix_ds = FoodSpectrumSet(
        x=mix_spec[None, :],
        wavenumbers=wn,
        metadata=pd.DataFrame({"sample_id": ["m1"]}),
        modality="raman",
    )
    mix_path = tmp_path / "mix.h5"
    create_library(mix_path, mix_ds)
    out_dir = tmp_path / "mix_reports"
    result = runner.invoke(
        app,
        [
            "mixture",
            str(mix_path),
            "--pure-hdf5",
            str(pure_path),
            "--spectrum-index",
            "0",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    _assert_report_created(out_dir)


def test_cli_hyperspectral(tmp_path: Path):
    height, width = 2, 2
    n_points = 10
    wn = np.linspace(1600, 1610, n_points)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.01, size=(height * width, n_points))
    ds = FoodSpectrumSet(
        x=X,
        wavenumbers=wn,
        metadata=pd.DataFrame({"sample_id": [f"p{i}" for i in range(height * width)]}),
        modality="raman",
    )
    flat_path = tmp_path / "flat.h5"
    create_library(flat_path, ds)
    out_dir = tmp_path / "hyper_reports"
    result = runner.invoke(
        app,
        [
            "hyperspectral",
            str(flat_path),
            "--height",
            str(height),
            "--width",
            str(width),
            "--target-wavenumber",
            "1605",
            "--window",
            "2",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    _assert_report_created(out_dir)


