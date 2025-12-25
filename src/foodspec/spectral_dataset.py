"""
Deprecated shim for SpectralDataset utilities.

Use foodspec.core.spectral_dataset instead. This module will be removed in a future release.
"""

from __future__ import annotations

import warnings

from foodspec.core.spectral_dataset import *  # noqa: F401,F403

warnings.warn(  # noqa: E402
    "foodspec.spectral_dataset is deprecated; use foodspec.core.spectral_dataset instead.",
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
