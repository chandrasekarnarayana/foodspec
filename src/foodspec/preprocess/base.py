"""Shared utilities for preprocessing transformers."""
from __future__ import annotations


from typing import Optional, TypeVar

import numpy as np

T = TypeVar("T", bound="WavenumberAwareMixin")


class WavenumberAwareMixin:
    """Mixin that stores a wavenumber axis for transformers that need it."""

    wavenumbers_: Optional[np.ndarray] = None

    def set_wavenumbers(self: T, wavenumbers: np.ndarray) -> T:
        self.wavenumbers_ = np.asarray(wavenumbers, dtype=float)
        return self

    def _assert_wavenumbers_set(self) -> None:
        if self.wavenumbers_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires wavenumbers; "
                "call set_wavenumbers(...) or fit(..., wavenumbers=...) first."
            )
