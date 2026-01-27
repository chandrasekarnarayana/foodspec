from __future__ import annotations

"""QC: Spectrum health scoring utilities.

Provides a lightweight scaffolding for computing per-spectrum health
scores (SNR, baseline drift, spike count, etc.) to support QC workflows.

This module defines `SpectrumHealthScore`, a dataclass responsible for:
- Validating inputs (grid alignment, modality hints)
- Producing a score table with placeholders for future algorithms
- Converting configuration to JSON-friendly dicts

Notes
-----
This is a scaffold only. Algorithms and thresholds are TODO.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class SpectrumHealthScore:
    """Compute basic health scores for spectra.

    Parameters
    ----------
    modality : str
        Spectroscopy modality (e.g., 'raman', 'ftir', 'nir'). Used for hints.
    reference_grid : Optional[np.ndarray]
        Target wavenumber grid for consistency checks.
    thresholds : Dict[str, float]
        Optional thresholds for flagging issues (placeholders).

    Methods
    -------
    score(data)
        Return a DataFrame with columns like 'snr', 'baseline_drift', 'spike_count', 'flags'.
    validate()
        Validate configuration and basic input integrity.
    to_dict()
        JSON-friendly representation of configuration.
    __hash__()
        Hash of the configuration for reproducibility.
    """

    modality: str = "raman"
    reference_grid: Optional[np.ndarray] = None
    thresholds: Dict[str, float] = field(default_factory=dict)

    def score(self, data: Any) -> pd.DataFrame:
        """Compute placeholder health scores for a batch of spectra.

        Parameters
        ----------
        data : FoodSpectrumSet | pd.DataFrame | np.ndarray
            Input spectral data. For arrays, shape should be (n_samples, n_points).

        Returns
        -------
        pd.DataFrame
            Table with columns ['snr', 'baseline_drift', 'spike_count', 'flags'].

        Notes
        -----
        - This is a scaffold. Replace zeros with actual computations.
        - Future implementations should estimate SNR, baseline drift, and spike counts.
        """
        # TODO: Implement actual computations (SNR, baseline drift, spike detection)
        n = 0
        if isinstance(data, np.ndarray):
            n = int(data.shape[0])
        elif isinstance(data, pd.DataFrame):
            n = int(len(data))
        else:
            # Fallback: try to get length from object with metadata
            try:
                n = int(len(getattr(data, "metadata", [])))
            except Exception:
                n = 0

        df = pd.DataFrame(
            {
                "snr": np.zeros(n, dtype=float),
                "baseline_drift": np.zeros(n, dtype=float),
                "spike_count": np.zeros(n, dtype=int),
                "flags": ["" for _ in range(n)],
            }
        )
        return df

    def validate(self) -> Dict[str, Any]:
        """Validate configuration and basic invariants.

        Returns
        -------
        Dict[str, Any]
            Validation report with 'ok' and 'issues'.
        """
        issues = []
        if self.reference_grid is not None:
            if not isinstance(self.reference_grid, np.ndarray):
                issues.append("reference_grid must be a numpy array if provided")
            elif self.reference_grid.ndim != 1:
                issues.append("reference_grid must be 1-D")
        if self.modality not in {"raman", "ftir", "nir"}:
            issues.append(f"Unknown modality: {self.modality}")
        return {"ok": len(issues) == 0, "issues": issues}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-friendly dict."""
        return {
            "modality": self.modality,
            "reference_grid_len": int(len(self.reference_grid)) if self.reference_grid is not None else None,
            "thresholds": dict(self.thresholds),
        }

    def __hash__(self) -> int:
        """Stable hash based on configuration."""
        key = (
            self.modality,
            tuple(self.reference_grid) if self.reference_grid is not None else None,
            tuple(sorted(self.thresholds.items())),
        )
        return hash(key)


__all__ = ["SpectrumHealthScore"]
