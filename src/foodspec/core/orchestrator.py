from __future__ import annotations

"""
Canonical orchestrator module for foodspec.core.

This module provides the public API for execution orchestration.
It re-exports from the protocol subsystem for a clean, canonical location.
"""


from foodspec.protocol.runner import ProtocolRunner as ExecutionEngine

__all__ = [
    "ExecutionEngine",
]
