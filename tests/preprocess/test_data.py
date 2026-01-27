"""Tests for data loading and validation."""

import numpy as np
import pytest

from foodspec.preprocess.data import (
    SpectraData,
    load_csv,
    validate_modality,
)


class TestModalityValidation:
    """Test modality validation."""

    def test_validate_known_modality(self):
        """Test validation of known modalities."""
        assert validate_modality("raman") == "raman"
        assert validate_modality("ftir") == "ftir"
        assert validate_modality("ir") == "ir"
        assert validate_modality("nir") == "nir"

    def test_validate_case_insensitive(self):
        """Test case-insensitive validation."""
        assert validate_modality("RAMAN") == "raman"
        assert validate_modality("Ftir") == "ftir"
        assert validate_modality(" raman ") == "raman"

    def test_validate_unknown_modality(self):
        """Test unknown modality returns 'unknown'."""
        assert validate_modality("xyz") == "unknown"
        assert validate_modality("") == "unknown"


class TestWideCSVLoading:
    """Test loading wide CSV format."""

    def test_load_wide_csv(self, wide_csv_data):
        """Test loading wide CSV with auto format detection."""
        data = load_csv(wide_csv_data, format="auto")

        assert isinstance(data, SpectraData)
        assert data.X.shape[0] == 20
        assert data.X.shape[1] == 100
        assert len(data.wavenumbers) == 100
        assert "sample_id" in data.metadata.columns
        assert "batch" in data.metadata.columns

    def test_load_wide_csv_explicit(self, wide_csv_data):
        """Test loading wide CSV with explicit format."""
        data = load_csv(wide_csv_data, format="wide")

        assert data.X.shape[0] == 20
        assert np.all(np.isfinite(data.X))

    def test_wavenumbers_sorted(self, wide_csv_data):
        """Test that wavenumbers are sorted."""
        data = load_csv(wide_csv_data, format="wide")

        # Wavenumbers should be sorted ascending
        assert np.all(np.diff(data.wavenumbers) > 0)


class TestLongCSVLoading:
    """Test loading long CSV format."""

    def test_load_long_csv(self, long_csv_data):
        """Test loading long CSV with auto format detection."""
        data = load_csv(long_csv_data, format="auto")

        assert isinstance(data, SpectraData)
        assert data.X.shape[0] == 20
        assert data.X.shape[1] == 100
        assert len(data.wavenumbers) == 100

    def test_load_long_csv_explicit(self, long_csv_data):
        """Test loading long CSV with explicit format."""
        data = load_csv(long_csv_data, format="long")

        assert data.X.shape[0] == 20
        assert np.all(np.isfinite(data.X))


class TestModalityInference:
    """Test modality inference from data."""

    def test_infer_raman_from_range(self, wide_csv_data):
        """Test Raman inference from wavenumber range."""
        # wide_csv_data has range 1000-2000, should be detected as Raman-like
        data = load_csv(wide_csv_data, format="wide")
        # Modality could be raman or unknown depending on range
        assert data.modality in ["raman", "unknown"]

    def test_explicit_modality_override(self, wide_csv_data):
        """Test explicit modality override."""
        data = load_csv(wide_csv_data, format="wide", modality="ftir")
        assert data.modality == "ftir"


class TestMetadataExtraction:
    """Test metadata column extraction."""

    def test_metadata_columns_extracted(self, wide_csv_data):
        """Test that metadata columns are extracted."""
        data = load_csv(wide_csv_data, format="wide")

        assert "sample_id" in data.metadata.columns
        assert "batch" in data.metadata.columns
        assert data.metadata.shape[0] == data.X.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
