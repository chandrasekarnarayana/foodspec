"""Data objects for FoodSpec (mindmap-aligned namespace)."""
from __future__ import annotations

from .metadata import validate_metadata
from .protocols import ProtocolConfig, ProtocolRunResult, ProtocolRunner, load_protocol, validate_protocol
from .spectrum import Spectrum
from .spectraset import FoodSpectrumSet, SpectraSet

__all__ = [
    "Spectrum",
    "SpectraSet",
    "FoodSpectrumSet",
    "ProtocolConfig",
    "ProtocolRunResult",
    "ProtocolRunner",
    "load_protocol",
    "validate_protocol",
    "validate_metadata",
]

