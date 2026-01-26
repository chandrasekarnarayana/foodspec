"""Instrument loaders (mindmap-aligned)."""
from __future__ import annotations

from .csv import load_csv_spectra
from .folder import load_folder, load_from_metadata_table
from .ftir import load_ftir_folder
from .raman import load_raman_folder
from .vendor_adapters import read_opus, read_spc

__all__ = [
    "load_csv_spectra",
    "load_folder",
    "load_from_metadata_table",
    "load_ftir_folder",
    "load_raman_folder",
    "read_opus",
    "read_spc",
]
