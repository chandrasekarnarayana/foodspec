"""Raman loader wrappers."""
from __future__ import annotations

from foodspec.io.loaders import load_folder


def load_raman_folder(*args, **kwargs):
    """Load Raman spectra from a folder of text files."""

    return load_folder(*args, modality="raman", **kwargs)


__all__ = ["load_raman_folder"]

