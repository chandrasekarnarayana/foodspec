"""
Deprecated shim for protocol engine.

Use foodspec.protocol instead. This module is retained for backward
compatibility and will be removed in a future release.
"""

from __future__ import annotations

import warnings

from foodspec.protocol import (
    EXAMPLE_PROTOCOL,
    ProtocolConfig,
    ProtocolRunner,
    ProtocolRunResult,
    list_available_protocols,
    load_protocol,
    validate_protocol,
)
from foodspec.protocol.steps import (
    STEP_REGISTRY,
    HarmonizeStep,
    HSIRoiStep,
    HSISegmentStep,
    OutputStep,
    PreprocessStep,
    QCStep,
    RQAnalysisStep,
    Step,
)

warnings.warn(
    "foodspec.protocol_engine is deprecated; use foodspec.protocol instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ProtocolConfig",
    "ProtocolRunResult",
    "ProtocolRunner",
    "Step",
    "PreprocessStep",
    "RQAnalysisStep",
    "OutputStep",
    "HarmonizeStep",
    "HSISegmentStep",
    "HSIRoiStep",
    "QCStep",
    "STEP_REGISTRY",
    "list_available_protocols",
    "load_protocol",
    "validate_protocol",
    "EXAMPLE_PROTOCOL",
]
