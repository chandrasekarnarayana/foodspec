from __future__ import annotations

"""
Deprecated shim for SpectralDataset utilities.

Use foodspec.data_objects.spectral_dataset instead. This module will be removed in a future release.
"""

"""
spectral_dataset - DEPRECATED

.. deprecated:: 1.1.0
    This module is deprecated and will be removed in v2.0.0.
    Use foodspec.data_objects.SpectralDataset instead.

This module is maintained for backward compatibility only.
All new code should use the modern API.

Migration Guide:
    Old: from foodspec.spectral_dataset import ...
    New: from foodspec.data_objects.SpectralDataset import ...

See: docs/migration/v1-to-v2.md
"""

import warnings

warnings.warn(
    "foodspec.spectral_dataset is deprecated and will be removed in v2.0.0. "
    "Use foodspec.data_objects.SpectralDataset instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Original module content continues below...
# ==============================================





import warnings

from foodspec.data_objects.spectral_dataset import *  # noqa: F401,F403

warnings.warn(  # noqa: E402
    "foodspec.spectral_dataset is deprecated; use foodspec.data_objects.spectral_dataset instead.",
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
