"""Backward compatibility shim for artifact.py (moved to deploy/artifact.py).

DEPRECATED: This module has been moved to foodspec.deploy.artifact.
Import from foodspec.deploy instead:
    from foodspec.deploy import save_artifact, load_artifact, Predictor

This shim will be removed in v2.0.0.
"""

import warnings

from foodspec.deploy.artifact import Predictor, load_artifact, save_artifact

warnings.warn(
    "Importing from foodspec.artifact is deprecated. "
    "Use 'from foodspec.deploy import save_artifact, load_artifact' instead. "
    "This backward-compatibility shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["save_artifact", "load_artifact", "Predictor"]
