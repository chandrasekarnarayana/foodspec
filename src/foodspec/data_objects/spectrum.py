"""Core Spectrum object: single spectrum with validated metadata and provenance."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

import numpy as np

SpectrumKind = Literal["raman", "ftir", "nir"]
AxisUnit = Literal["cm-1", "nm", "um", "1/cm"]


def _validate_spectrum_schema(metadata: Dict[str, Any]) -> None:
    """Validate spectrum metadata against expected schema.

    Args:
        metadata (dict): Metadata dictionary to validate.

    Raises:
        TypeError: If ``metadata`` is not a dict.
        ValueError: If optional fields have incorrect types.
    """
    # Allow any metadata; basic checks only
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    # Optional field checks
    if "sample_id" in metadata and not isinstance(metadata["sample_id"], str):
        raise ValueError("metadata['sample_id'] must be string if present")


@dataclass
class Spectrum:
    """Single spectrum with axis, intensity, units, kind, and metadata.

    Represents a single spectroscopic measurement with provenance tracking.

    Args:
        x (np.ndarray): X-axis (wavenumber/wavelength), shape (n_points,).
        y (np.ndarray): Intensity values, shape (n_points,).
        kind (Literal['raman','ftir','nir']): Spectroscopy modality.
        x_unit (Literal['cm-1','nm','um','1/cm']): Axis unit. Default 'cm-1'.
        metadata (dict): Optional metadata (sample_id, instrument, etc.).

    Attributes:
        x (np.ndarray): X-axis data.
        y (np.ndarray): Y-axis data.
        kind (str): Modality.
        x_unit (str): Unit of x-axis.
        metadata (dict): Validated metadata.
        config_hash (str): Hash of metadata for reproducibility tracking.
    """

    x: np.ndarray
    y: np.ndarray
    kind: SpectrumKind
    x_unit: AxisUnit = "cm-1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize inputs (shapes, modality, metadata)."""
        # Convert to numpy arrays
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)

        # Validate shapes
        if self.x.ndim != 1:
            raise ValueError(f"x must be 1D; got shape {self.x.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"x and y must have same length; got {self.x.shape[0]} vs {self.y.shape[0]}")

        # Validate kind
        if self.kind not in ("raman", "ftir", "nir"):
            raise ValueError(f"kind must be 'raman', 'ftir', or 'nir'; got {self.kind}")

        # Validate and ensure metadata is dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        _validate_spectrum_schema(self.metadata)

    @property
    def config_hash(self) -> str:
        """Hash of metadata for reproducibility tracking.

        Returns:
            str: First 8 hex chars of SHA256 over metadata JSON.
        """
        import json

        meta_str = json.dumps(self.metadata, sort_keys=True, default=str)
        return hashlib.sha256(meta_str.encode()).hexdigest()[:8]

    @property
    def n_points(self) -> int:
        """Number of spectral points.

        Returns:
            int: Length of ``x``/``y``.
        """
        return len(self.x)

    def copy(self) -> Spectrum:
        """Return a deep copy of this spectrum.

        Returns:
            Spectrum: Independent copy.
        """
        return Spectrum(
            x=self.x.copy(),
            y=self.y.copy(),
            kind=self.kind,
            x_unit=self.x_unit,
            metadata={k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.metadata.items()},
        )

    def crop_wavenumber(self, x_min: float, x_max: float) -> Spectrum:
        """Crop spectrum to a wavenumber/wavelength range.

        Args:
            x_min (float): Minimum axis value.
            x_max (float): Maximum axis value.

        Returns:
            Spectrum: New spectrum with cropped data.

        Raises:
            ValueError: If the range contains no points.
        """
        mask = (self.x >= x_min) & (self.x <= x_max)
        if not np.any(mask):
            raise ValueError(f"No data in range [{x_min}, {x_max}]")
        return Spectrum(
            x=self.x[mask],
            y=self.y[mask],
            kind=self.kind,
            x_unit=self.x_unit,
            metadata=self.metadata.copy(),
        )

    def normalize(self, method: str = "vector") -> Spectrum:
        """Normalize spectrum.

        Args:
            method (str): One of "vector", "max", or "area".

        Returns:
            Spectrum: Normalized spectrum.

        Raises:
            ValueError: If method is unknown.
        """
        y_norm = self.y.copy()

        if method == "vector":
            norm = np.linalg.norm(y_norm)
            if norm > 0:
                y_norm = y_norm / norm
        elif method == "max":
            max_val = np.max(np.abs(y_norm))
            if max_val > 0:
                y_norm = y_norm / max_val
        elif method == "area":
            area = np.trapz(np.abs(y_norm), self.x)
            if area > 0:
                y_norm = y_norm / area
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return Spectrum(
            x=self.x.copy(),
            y=y_norm,
            kind=self.kind,
            x_unit=self.x_unit,
            metadata=self.metadata.copy(),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Spectrum({self.kind}, n={self.n_points}, x_range=[{self.x.min():.1f}, {self.x.max():.1f}] {self.x_unit})"
        )
