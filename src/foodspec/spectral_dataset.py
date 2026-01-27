"""Deprecated shim for SpectralDataset utilities.

Use foodspec.data_objects.spectral_dataset instead. This module will be removed in a future release.
"""

from __future__ import annotations

import warnings

from foodspec.data_objects.spectral_dataset import *  # noqa: F401,F403

warnings.warn(
    "foodspec.spectral_dataset is deprecated and will be removed in v2.0.0. "
    "Use foodspec.data_objects.SpectralDataset instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [  # noqa: F405,F401
    "PreprocessingConfig",
    "PreprocessOptions",
    "SpectralDataset",
    "HyperspectralDataset",
    "harmonize_datasets",
    "HDF5_SCHEMA_VERSION",
    "baseline_rubberband",
    "baseline_als",
    "baseline_polynomial",
]
