"""Data loaders for foodspec."""

from foodspec.data.loader import load_example_oils
from foodspec.data.public import (
    load_public_evoo_sunflower_raman,
    load_public_ftir_oils,
    load_public_mendeley_oils,
)

__all__ = [
    "load_example_oils",
    "load_public_mendeley_oils",
    "load_public_evoo_sunflower_raman",
    "load_public_ftir_oils",
]
