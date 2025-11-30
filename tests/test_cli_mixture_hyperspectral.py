from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from foodspec.cli import app
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.data.libraries import create_library

runner = CliRunner()


def _make_mixture_hdf5(tmp_path: Path) -> tuple[Path, Path]:
    wn = np.linspace(1600, 1650, 20)
    comp1 = np.exp(-0.5 * ((wn - 1610) / 2) ** 2)
    comp2 = 0.5 * np.exp(-0.5 * ((wn - 1635) / 3) ** 2)
    pure_X = np.vstack([comp1, comp2])
    pure_meta = pd.DataFrame({"sample_id": ["c1", "c2"]})
    pure_ds = FoodSpectrumSet(x=pure_X, wavenumbers=wn, metadata=pure_meta, modality="raman")
    pure_path = tmp_path / "pure.h5"
    create_library(pure_path, pure_ds)

    coeffs = np.array([0.7, 0.3])
    mix_spec = coeffs @ pure_X
    mix_ds = FoodSpectrumSet(
        x=mix_spec[None, :],
        wavenumbers=wn,
        metadata=pd.DataFrame({"sample_id": ["m1"]}),
        modality="raman",
    )
    mix_path = tmp_path / "mix.h5"
    create_library(mix_path, mix_ds)
    return mix_path, pure_path


def _make_hyperspectral_hdf5(tmp_path: Path) -> Path:
    height, width = 2, 2
    n_points = 10
    wn = np.linspace(1600, 1610, n_points)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.01, size=(height * width, n_points))
    meta = pd.DataFrame({"sample_id": [f"p{i}" for i in range(height * width)]})
    ds = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")
    path = tmp_path / "cube_flat.h5"
    create_library(path, ds)
    return path


def _assert_report(report_root: Path):
    reports = list(report_root.iterdir())
    assert reports
    report_dir = reports[0]
    assert (report_dir / "summary.json").exists()


def test_cli_mixture(tmp_path):
    mix_path, pure_path = _make_mixture_hdf5(tmp_path)
    out_dir = tmp_path / "out_mix"
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
    _assert_report(out_dir)


def test_cli_hyperspectral(tmp_path):
    flat_path = _make_hyperspectral_hdf5(tmp_path)
    out_dir = tmp_path / "out_hyper"
    result = runner.invoke(
        app,
        [
            "hyperspectral",
            str(flat_path),
            "--height",
            "2",
            "--width",
            "2",
            "--target-wavenumber",
            "1605",
            "--window",
            "2",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    _assert_report(out_dir)
