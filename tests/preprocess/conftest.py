"""Test fixtures for preprocessing tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.data_objects.spectra_set import FoodSpectrumSet


@pytest.fixture
def synthetic_raman_data():
    """Generate synthetic Raman data.

    Returns 50 samples with 512 wavenumbers from 500-3000 cm-1.
    """
    n_samples = 50
    n_features = 512
    wavenumbers = np.linspace(500, 3000, n_features)

    # Generate synthetic spectra with peaks
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # Add baseline (polynomial)
        baseline = 0.1 + 0.001 * wavenumbers + 1e-7 * wavenumbers**2

        # Add peaks
        for peak_loc in [800, 1200, 1600, 2000, 2800]:
            peak_idx = np.argmin(np.abs(wavenumbers - peak_loc))
            peak_width = 20 + np.random.rand() * 10
            peak_height = 0.5 + np.random.rand() * 0.5
            peak = peak_height * np.exp(-0.5 * ((np.arange(n_features) - peak_idx) / peak_width) ** 2)
            X[i] += peak

        # Add baseline
        X[i] += baseline

        # Add noise
        X[i] += 0.02 * np.random.randn(n_features)

        # Add random spikes (cosmic rays)
        if i % 10 == 0:
            spike_idx = np.random.randint(0, n_features, size=2)
            X[i, spike_idx] += 2.0

    metadata = pd.DataFrame({
        "sample_id": [f"sample_{i}" for i in range(n_samples)],
        "batch": [f"B{i//10}" for i in range(n_samples)],
        "label": np.random.choice(["ClassA", "ClassB"], n_samples),
    })

    return FoodSpectrumSet(
        x=X,
        wavenumbers=wavenumbers,
        metadata=metadata,
        modality="raman",
    )


@pytest.fixture
def synthetic_ftir_data():
    """Generate synthetic FTIR data.

    Returns 50 samples with 512 wavenumbers from 400-4000 cm-1.
    """
    n_samples = 50
    n_features = 512
    wavenumbers = np.linspace(400, 4000, n_features)

    # Generate synthetic FTIR spectra
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # Add baseline
        baseline = 0.2 + 0.0001 * wavenumbers

        # Add characteristic FTIR peaks
        for peak_loc in [1000, 1500, 1700, 2900, 3300]:
            peak_idx = np.argmin(np.abs(wavenumbers - peak_loc))
            peak_width = 30 + np.random.rand() * 15
            peak_height = 0.6 + np.random.rand() * 0.4
            peak = peak_height * np.exp(-0.5 * ((np.arange(n_features) - peak_idx) / peak_width) ** 2)
            X[i] += peak

        # Add baseline
        X[i] += baseline

        # Add noise
        X[i] += 0.015 * np.random.randn(n_features)

    metadata = pd.DataFrame({
        "sample_id": [f"ftir_{i}" for i in range(n_samples)],
        "batch": [f"batch{i//10}" for i in range(n_samples)],
        "matrix": np.random.choice(["oil", "powder"], n_samples),
    })

    return FoodSpectrumSet(
        x=X,
        wavenumbers=wavenumbers,
        metadata=metadata,
        modality="ftir",
    )


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_recipe_dict():
    """Sample recipe dictionary."""
    return {
        "modality": "raman",
        "steps": [
            {"op": "despike", "window": 5, "threshold": 5.0},
            {"op": "baseline", "method": "als", "lam": 1e5, "p": 0.01},
            {"op": "smoothing", "method": "savgol", "window_length": 7, "polyorder": 3},
            {"op": "normalization", "method": "snv"},
        ],
    }


@pytest.fixture
def wide_csv_data(tmp_path):
    """Generate wide CSV format data."""
    n_samples = 20
    n_features = 100
    wavenumbers = np.linspace(1000, 2000, n_features)

    X = np.random.rand(n_samples, n_features)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"{wn:.1f}" for wn in wavenumbers])
    df.insert(0, "sample_id", [f"S{i}" for i in range(n_samples)])
    df.insert(1, "batch", [f"B{i//5}" for i in range(n_samples)])

    # Save to CSV
    csv_path = tmp_path / "wide_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def long_csv_data(tmp_path):
    """Generate long CSV format data."""
    n_samples = 20
    n_features = 100
    wavenumbers = np.linspace(1000, 2000, n_features)

    rows = []
    for i in range(n_samples):
        for j, wn in enumerate(wavenumbers):
            rows.append({
                "sample_id": f"S{i}",
                "batch": f"B{i//5}",
                "wavenumber": wn,
                "intensity": np.random.rand(),
            })

    df = pd.DataFrame(rows)

    # Save to CSV
    csv_path = tmp_path / "long_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path
