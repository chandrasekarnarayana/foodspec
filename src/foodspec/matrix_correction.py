"""
Backward-compatibility shim for matrix_correction module.

This module has been moved to foodspec.preprocess.matrix_correction.
This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "foodspec.matrix_correction is deprecated; use foodspec.preprocess.matrix_correction instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foodspec.preprocess.matrix_correction import *  # noqa: F401,F403,E402
