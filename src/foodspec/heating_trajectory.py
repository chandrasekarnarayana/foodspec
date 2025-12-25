"""
Backward-compatibility shim for heating_trajectory module.

This module has been moved to foodspec.workflows.heating_trajectory.
This shim will be removed in a future version.
"""
import warnings

warnings.warn(
    "foodspec.heating_trajectory is deprecated; use foodspec.workflows.heating_trajectory instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foodspec.workflows.heating_trajectory import *  # noqa: F401,F403,E402
