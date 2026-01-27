"""Hyperspectral cube support for Raman/FTIR maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from foodspec.data_objects.spectra_set import FoodSpectrumSet

__all__ = ["HyperSpectralCube"]


@dataclass
class HyperSpectralCube:
    """Container for hyperspectral maps stored as (height, width, n_points)."""

    cube: np.ndarray  # shape (height, width, n_points)
    wavenumbers: np.ndarray  # shape (n_points,)
    metadata: pd.DataFrame
    image_shape: Tuple[int, int]

    def __post_init__(self) -> None:
        self.cube = np.asarray(self.cube, dtype=float)
        self.wavenumbers = np.asarray(self.wavenumbers, dtype=float)
        if self.cube.ndim != 3:
            raise ValueError("cube must be 3D (height, width, n_points).")
        h, w, n_points = self.cube.shape
        if self.wavenumbers.shape[0] != n_points:
            raise ValueError("wavenumbers length must match cube spectral dimension.")
        if (h, w) != self.image_shape:
            raise ValueError("image_shape does not match cube spatial dimensions.")
        if not isinstance(self.metadata, pd.DataFrame):
            raise ValueError("metadata must be a pandas DataFrame.")

    @classmethod
    def from_spectrum_set(cls, spectra: FoodSpectrumSet, image_shape: Tuple[int, int]) -> "HyperSpectralCube":
        """Create cube from flattened spectra using image_shape (h, w)."""
        h, w = image_shape
        n_pixels = h * w
        if len(spectra) != n_pixels:
            raise ValueError("Number of spectra does not match image_shape pixels.")
        cube = spectra.x.reshape(h, w, -1)
        return cls(
            cube=cube,
            wavenumbers=spectra.wavenumbers,
            metadata=spectra.metadata.copy(),
            image_shape=image_shape,
        )

    def to_spectrum_set(self, modality: str = "raman") -> FoodSpectrumSet:
        """Flatten cube to FoodSpectrumSet, adding row/col coordinates."""
        h, w, n_points = self.cube.shape
        flat = self.cube.reshape(h * w, n_points)
        coords = pd.DataFrame({"row": np.repeat(np.arange(h), w), "col": np.tile(np.arange(w), h)})
        meta = self.metadata.copy().reset_index(drop=True)
        meta = pd.concat([coords, meta], axis=1)
        return FoodSpectrumSet(x=flat, wavenumbers=self.wavenumbers, metadata=meta, modality=modality)

    def to_pixel_spectra(self, modality: str = "raman") -> FoodSpectrumSet:
        """Flatten to pixel spectra with row/col metadata."""
        return self.to_spectrum_set(modality=modality)

    def from_pixel_labels(self, labels: np.ndarray) -> np.ndarray:
        """Reshape flat labels (n_pixels,) to (height, width) label image."""
        labels = np.asarray(labels)
        h, w = self.image_shape
        if labels.shape[0] != h * w:
            raise ValueError("labels length must match number of pixels (height*width).")
        return labels.reshape(h, w)

    def get_pixel_spectrum(self, row: int, col: int) -> np.ndarray:
        """Return spectrum at a given pixel coordinate."""
        if row < 0 or row >= self.image_shape[0] or col < 0 or col >= self.image_shape[1]:
            raise IndexError("Pixel indices out of range.")
        return self.cube[row, col, :]

    def mean_spectrum(self) -> np.ndarray:
        """Return mean spectrum over all pixels."""
        return self.cube.reshape(-1, self.cube.shape[-1]).mean(axis=0)
