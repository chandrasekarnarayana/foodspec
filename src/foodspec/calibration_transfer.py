"""
Backward-compatibility shim for calibration_transfer module.

This module has been moved to foodspec.preprocess.calibration_transfer.
This shim will be removed in a future version.
"""
import warnings

warnings.warn(
    "foodspec.calibration_transfer is deprecated; use foodspec.preprocess.calibration_transfer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foodspec.preprocess.calibration_transfer import *  # noqa: F401,F403,E402
