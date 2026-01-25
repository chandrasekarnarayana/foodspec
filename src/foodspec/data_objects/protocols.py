"""Protocol objects (shim to foodspec.protocol)."""
from __future__ import annotations

from foodspec.protocol import ProtocolConfig, ProtocolRunResult, ProtocolRunner, load_protocol, validate_protocol

__all__ = [
    "ProtocolConfig",
    "ProtocolRunResult",
    "ProtocolRunner",
    "load_protocol",
    "validate_protocol",
]

