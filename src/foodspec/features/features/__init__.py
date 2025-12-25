# Backward-compatibility shim for legacy 'foodspec.features.features' imports
# Re-export symbols from the canonical 'foodspec.features' package

from foodspec.features import *  # noqa: F401,F403

# Explicit re-exports for RQEngine and types if tests import nested path
try:
    from foodspec.features.rq import (
        PeakDefinition,
        RatioDefinition,
        RatioQualityEngine,
        RatioQualityResult,
        RQConfig,
    )
except Exception:  # pragma: no cover,noqa: BLE001
    # Allow import to succeed even if submodule missing in minimal installs
    pass
