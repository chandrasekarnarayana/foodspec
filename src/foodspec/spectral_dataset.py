"""Deprecated shim for SpectralDataset utilities.

Use foodspec.data_objects.spectral_dataset instead. This module will be removed in a future release.
"""

from __future__ import annotations

import warnings

# Import from new location
from foodspec.data_objects.spectral_dataset import (
    HDF5_SCHEMA_VERSION,
    HyperspectralDataset,
    PreprocessingConfig,
    SpectralDataset,
    baseline_als,
    baseline_polynomial,
    baseline_rubberband,
    harmonize_datasets,
)

__all__ = [
    "PreprocessingConfig",
    "SpectralDataset",
    "HyperspectralDataset",
    "harmonize_datasets",
    "HDF5_SCHEMA_VERSION",
    "baseline_rubberband",
    "baseline_als",
    "baseline_polynomial",
]

# Emit deprecation warning when this module is imported
warnings.warn(
    "foodspec.spectral_dataset is deprecated and will be removed in v2.0.0. "
    "Use foodspec.data_objects.spectral_dataset instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)
