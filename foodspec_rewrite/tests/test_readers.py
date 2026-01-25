"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
"""

from pathlib import Path

import pandas as pd
import pytest

from foodspec.core.protocol import DataSpec
from foodspec.io import load_csv_spectra


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_csv_spectra_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    rows = [
        {"id": "a", "mod": "raman", "label": 1, "1000": 1.0, "1001": 2.0},
        {"id": "b", "mod": "raman", "label": 0, "1000": 3.0, "1001": 4.0},
    ]
    _write_csv(csv_path, rows)

    spec = DataSpec(
        input=str(csv_path),
        modality="raman",
        label="label",
        metadata_map={"sample_id": "id", "modality": "mod", "label": "label"},
    )

    ds = load_csv_spectra(csv_path, spec)
    assert ds.X.shape == (2, 2)
    assert ds.x.tolist() == [1000.0, 1001.0]
    assert ds.y.tolist() == [1, 0]
    assert "sample_id" in ds.metadata.columns
    assert ds.metadata.loc[0, "sample_id"] == "a"


def test_load_csv_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    rows = [{"id": "a", "label": 1, "1000": 1.0}]
    _write_csv(csv_path, rows)

    spec = DataSpec(
        input=str(csv_path),
        modality="raman",
        label="label",
        metadata_map={"sample_id": "id", "modality": "mod"},
    )

    with pytest.raises(ValueError) as err:
        load_csv_spectra(csv_path, spec)

    assert "Missing required columns" in str(err.value)


def test_load_csv_non_numeric_wavenumber_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    rows = [{"id": "a", "label": 1, "foo": 1.0}]
    _write_csv(csv_path, rows)

    spec = DataSpec(
        input=str(csv_path),
        modality="raman",
        label="label",
        metadata_map={"sample_id": "id"},
    )

    with pytest.raises(ValueError) as err:
        load_csv_spectra(csv_path, spec)

    assert "Wavenumber columns" in str(err.value)
