from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from foodspec.cli import app
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.data.libraries import create_library

runner = CliRunner()


def _make_oil_hdf5(tmp_path: Path, fname: str) -> Path:
    """Create a small HDF5 library for oil-auth CLI testing."""

    wavenumbers = np.linspace(1600.0, 1800.0, 50)
    n_per_class = 5
    rng = np.random.default_rng(0)
    class_a = np.array(
        [np.exp(-0.5 * ((wavenumbers - 1655) / 5) ** 2) + rng.normal(0, 0.01, size=wavenumbers.shape)] * n_per_class
    )
    class_b = np.array(
        [0.5 * np.exp(-0.5 * ((wavenumbers - 1742) / 6) ** 2) + rng.normal(0, 0.01, size=wavenumbers.shape)]
        * n_per_class
    )
    X = np.vstack([class_a, class_b])
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(X.shape[0])],
            "oil_type": ["A"] * n_per_class + ["B"] * n_per_class,
        }
    )
    ds = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=metadata, modality="raman")
    path = tmp_path / fname
    create_library(path, ds)
    return path


def test_cli_oil_auth_save_and_model_info(tmp_path):
    hdf5_path = _make_oil_hdf5(tmp_path, "oil.h5")
    model_base = tmp_path / "model_test"
    report_path = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "oil-auth",
            str(hdf5_path),
            "--output-report",
            str(report_path),
            "--save-model",
            str(model_base),
        ],
    )
    assert result.exit_code == 0, result.output
    assert model_base.with_suffix(".joblib").exists()
    assert model_base.with_suffix(".json").exists()

    info_result = runner.invoke(app, ["model-info", str(model_base)])
    assert info_result.exit_code == 0, info_result.output
    assert "Name:" in info_result.stdout
    assert "Version" in info_result.stdout
