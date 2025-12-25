"""Phase 2 ingestion/standardization tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from foodspec.io.ingest import DEFAULT_IO_REGISTRY, IORegistry, load_csv_or_txt, load_folder_pattern


def _write_csv(path: Path, wavenumbers: np.ndarray, spectra: np.ndarray) -> None:
    df = pd.DataFrame(np.column_stack([wavenumbers, spectra.T]))
    df.to_csv(path, index=False)


def test_csv_loader_infers_delimiter_and_metrics():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wn = np.linspace(500, 600, 5)
        spectra = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
            ]
        )
        csv_path = tmp_path / "spectra.csv"
        _write_csv(csv_path, wn, spectra)

        result = load_csv_or_txt(csv_path)
        ds = result.dataset

        assert ds.x.shape == (2, 5)
        assert result.metrics["parsed_pct"] == 100.0
        assert result.metrics["monotonic_axis"] is True
        assert result.metrics["grid_nonuniformity"] < 1e-6


def test_folder_loader_handles_resampling_and_metrics():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wn1 = np.array([500, 510, 520, 530])
        wn2 = np.array([500, 512, 524, 536])
        spec1 = np.array([[1, 2, 3, 4]])
        spec2 = np.array([[2, 3, 4, 5]])

        _write_csv(tmp_path / "a.csv", wn1, spec1)
        _write_csv(tmp_path / "b.csv", wn2, spec2)

        result = load_folder_pattern(tmp_path, pattern="*.csv")
        metrics = result.metrics

        # Two files parsed, one resampled
        assert metrics["parsed_files"] == 2
        assert metrics["resampled_spectra"] >= 1
        assert metrics["parsed_pct"] == 100.0
        assert metrics["resampled_pct"] > 0
        assert metrics["monotonic_axis"] is True


def test_registry_auto_loader_uses_default_registry():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wn = np.linspace(700, 800, 4)
        spec = np.array([[1, 1, 1, 1]])
        csv_path = tmp_path / "single.csv"
        _write_csv(csv_path, wn, spec)

        registry = IORegistry()
        registry.register("csv", load_csv_or_txt)

        result = registry.load("csv", csv_path)
        assert result.dataset.x.shape[0] == 1

        # Default registry should also load via auto
        default_result = DEFAULT_IO_REGISTRY.load("auto", csv_path)
        assert default_result.dataset.x.shape == (1, 4)
        assert default_result.metrics["parsed_pct"] == 100.0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
