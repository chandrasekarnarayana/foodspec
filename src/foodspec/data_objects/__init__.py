"""Data objects for FoodSpec (mindmap-aligned namespace)."""

from __future__ import annotations

from .metadata import validate_metadata
from .protocols import ProtocolConfig, ProtocolRunner, ProtocolRunResult, load_protocol, validate_protocol
from .spectra_set import FoodSpectrumSet, SpectraSet
from .spectral_dataset import (
    HDF5_SCHEMA_VERSION,
    HyperspectralDataset,
    PreprocessingConfig,
    SpectralDataset,
)
from .spectrum import Spectrum

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
    "HDF5_SCHEMA_VERSION",
    "HyperspectralDataset",
    "PreprocessingConfig",
    "SpectralDataset",
]
