"""
Backward-compatibility shim for cli.py module.

This module has been moved to foodspec.cli.main.
This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "foodspec.cli is deprecated; use foodspec.cli.main instead.",
    DeprecationWarning,
    stacklevel=2,
)

from foodspec.cli.main import *  # noqa: F401,F403,E402
