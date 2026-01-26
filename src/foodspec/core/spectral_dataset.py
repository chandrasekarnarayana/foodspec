"""Legacy import path for SpectralDataset (deprecated)."""
from __future__ import annotations

from foodspec.utils.deprecation import warn_deprecated_import
from foodspec.data_objects.spectral_dataset import (
    HDF5_SCHEMA_VERSION,
    HyperspectralDataset,
    PreprocessingConfig,
    SpectralDataset,
    baseline_als,
    baseline_polynomial,
    baseline_rubberband,
    harmonize_datasets,
    remove_spikes,
    smooth_signal,
    normalize_matrix,
)

warn_deprecated_import("foodspec.core.spectral_dataset", "foodspec.data_objects.spectral_dataset")

__all__ = [
    "HDF5_SCHEMA_VERSION",
    "HyperspectralDataset",
    "PreprocessingConfig",
    "SpectralDataset",
    "baseline_als",
    "baseline_polynomial",
    "baseline_rubberband",
    "harmonize_datasets",
    "remove_spikes",
    "smooth_signal",
    "normalize_matrix",
]

