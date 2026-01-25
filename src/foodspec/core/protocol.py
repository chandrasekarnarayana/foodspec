from __future__ import annotations
"""
Canonical protocol module for foodspec.core.

This module provides the public API for protocol execution and management.
It re-exports from the protocol subsystem for a clean, canonical location.
"""


from foodspec.protocol.runner import ProtocolRunner as ProtocolV2
from foodspec.protocol.config import ProtocolConfig, ProtocolRunResult

__all__ = [
    "ProtocolV2",
    "ProtocolConfig",
    "ProtocolRunResult",
]
