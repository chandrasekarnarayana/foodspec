"""Deprecated leakage shim for foodspec.validation."""

from __future__ import annotations

from foodspec.qc.leakage import (
    detect_batch_label_correlation,
    detect_leakage,
    detect_replicate_leakage,
)
from foodspec.utils.deprecation import warn_deprecated_import

warn_deprecated_import("foodspec.validation.leakage", "foodspec.qc.leakage")

__all__ = [
    "detect_batch_label_correlation",
    "detect_leakage",
    "detect_replicate_leakage",
]
