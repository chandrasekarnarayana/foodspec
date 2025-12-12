"""
Advanced harmonization utilities:
- Instrument-specific calibration curves for wavenumber drift correction.
- Intensity normalization using metadata (e.g., laser power).
- Diagnostics (pre/post alignment metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from foodspec.spectral_dataset import SpectralDataset


@dataclass
class CalibrationCurve:
    instrument_id: str
    wn_source: np.ndarray
    wn_target: np.ndarray

    def apply(self, wn: np.ndarray) -> np.ndarray:
        return np.interp(wn, self.wn_source, self.wn_target)


def apply_calibration(ds: SpectralDataset, curve: CalibrationCurve) -> SpectralDataset:
    wn_corr = curve.apply(ds.wavenumbers)
    ds_corr = ds.copy()
    ds_corr.wavenumbers = wn_corr
    ds_corr.logs.append(f"calibrated_with={curve.instrument_id}")
    ds_corr.history.append({"step": "calibration", "instrument": curve.instrument_id})
    return ds_corr


def intensity_normalize_by_power(ds: SpectralDataset, power_mw: Optional[float]) -> SpectralDataset:
    if power_mw is None or power_mw <= 0:
        return ds
    ds_norm = ds.copy()
    ds_norm.spectra = ds_norm.spectra / power_mw
    ds_norm.logs.append(f"intensity_normalized_by_power={power_mw}mW")
    ds_norm.history.append({"step": "intensity_norm_power", "power_mw": power_mw})
    return ds_norm


def harmonize_datasets_advanced(
    datasets: List[SpectralDataset],
    calibration_curves: Dict[str, CalibrationCurve],
    intensity_meta_key: str = "laser_power_mw",
) -> Tuple[List[SpectralDataset], Dict[str, float]]:
    """
    Apply calibration curves and power normalization, then interpolate to a common grid.
    Returns harmonized datasets + diagnostics (residual std dev).
    """
    calibrated = []
    for ds in datasets:
        inst_id = ds.instrument_meta.get("instrument_id", "unknown")
        curve = calibration_curves.get(inst_id)
        tmp = apply_calibration(ds, curve) if curve else ds
        power = ds.instrument_meta.get(intensity_meta_key)
        tmp = intensity_normalize_by_power(tmp, power)
        calibrated.append(tmp)

    # Common grid from median wavenumbers
    target_grid = max(calibrated, key=lambda d: len(d.wavenumbers)).wavenumbers
    harmonized = []
    for ds in calibrated:
        wn = ds.wavenumbers
        X = ds.spectra
        interp = np.vstack([np.interp(target_grid, wn, row) for row in X])
        ds_h = ds.copy()
        ds_h.wavenumbers = target_grid.copy()
        ds_h.spectra = interp
        ds_h.logs.append(f"advanced_harmonized_to_grid len={len(target_grid)}")
        ds_h.history.append({"step": "advanced_harmonize", "len_grid": len(target_grid)})
        ds_h.instrument_meta.setdefault("harmonization_params", {})
        ds_h.instrument_meta["harmonization_params"]["target_grid_len"] = len(target_grid)
        harmonized.append(ds_h)

    # Diagnostics: residual variation across instruments for overlapping samples
    diag = {}
    all_spec = np.concatenate([h.spectra for h in harmonized], axis=0)
    diag["residual_std_mean"] = float(np.nanmean(np.std(all_spec, axis=0)))
    return harmonized, diag


def plot_harmonization_diagnostics(datasets: List[SpectralDataset]):
    fig = plt.figure(figsize=(6, 4))
    for ds in datasets:
        label = ds.instrument_meta.get("instrument_id", "inst")
        plt.plot(ds.wavenumbers, np.nanmean(ds.spectra, axis=0), alpha=0.5, label=label)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Mean intensity (a.u.)")
    plt.legend()
    plt.tight_layout()
    return fig
