"""FTIR loader wrappers."""

from __future__ import annotations

from foodspec.io.loaders.folder import load_folder


def load_ftir_folder(*args, **kwargs):
    """Load FTIR spectra from a folder of text files."""

    return load_folder(*args, modality="ftir", **kwargs)


__all__ = ["load_ftir_folder"]
