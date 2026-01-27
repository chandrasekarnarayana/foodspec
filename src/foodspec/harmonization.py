from __future__ import annotations

"""
Advanced harmonization utilities:
- Instrument-specific calibration curves for wavenumber drift correction.
- Intensity normalization using metadata (e.g., laser power).
- Diagnostics (pre/post alignment metrics).
"""

"""
harmonization - DEPRECATED

.. deprecated:: 1.1.0
    This module is deprecated and will be removed in v2.0.0.
    Use foodspec.core.harmonize_datasets instead.

This module is maintained for backward compatibility only.
All new code should use the modern API.

Migration Guide:
    Old: from foodspec.harmonization import ...
    New: from foodspec.core.harmonize_datasets import ...

See: docs/migration/v1-to-v2.md
"""

import warnings

warnings.warn(
    "foodspec.harmonization is deprecated and will be removed in v2.0.0. "
    "Use foodspec.core.harmonize_datasets instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Original module content continues below...
# ==============================================





from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from foodspec.data_objects.spectral_dataset import SpectralDataset


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


def estimate_calibration_curve(
    reference: SpectralDataset,
    target: SpectralDataset,
    *,
    max_shift_points: int = 25,
) -> Tuple[CalibrationCurve, Dict[str, float]]:
    """Estimate a calibration curve by maximizing mean-spectrum correlation vs. the reference.

    Uses integer point shifts (±max_shift_points) on the target mean spectrum to find the best
    correlation with the reference mean, then maps wavenumbers via the inferred shift.
    """

    ref_mean = np.nanmean(reference.spectra, axis=0)
    tgt_mean = np.nanmean(target.spectra, axis=0)
    if ref_mean.shape[0] != tgt_mean.shape[0]:
        raise ValueError("Reference and target spectra must share the same length for curve estimation.")

    wn_step = float(np.median(np.diff(reference.wavenumbers)))
    best_shift = 0
    best_corr = -np.inf
    for shift in range(-max_shift_points, max_shift_points + 1):
        shifted = np.roll(tgt_mean, shift)
        corr = np.corrcoef(ref_mean, shifted)[0, 1]
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr = float(corr)
            best_shift = shift

    shift_cm = best_shift * wn_step
    wn_target = target.wavenumbers + shift_cm
    curve = CalibrationCurve(
        instrument_id=target.instrument_meta.get("instrument_id", "unknown"),
        wn_source=target.wavenumbers,
        wn_target=wn_target,
    )
    diag = {"best_shift_points": int(best_shift), "shift_cm": float(shift_cm), "corr_coeff": float(best_corr)}
    return curve, diag


def generate_calibration_curves(
    datasets: List[SpectralDataset], reference_instrument_id: str, *, max_shift_points: int = 25
) -> Tuple[Dict[str, CalibrationCurve], Dict[str, Dict[str, float]]]:
    """Generate calibration curves for all datasets relative to the reference instrument.

    Returns (curves, diagnostics) keyed by instrument_id.
    """

    ref = None
    for ds in datasets:
        if ds.instrument_meta.get("instrument_id") == reference_instrument_id:
            ref = ds
            break
    if ref is None:
        raise ValueError(f"reference_instrument_id '{reference_instrument_id}' not found in datasets")

    curves: Dict[str, CalibrationCurve] = {}
    diagnostics: Dict[str, Dict[str, float]] = {}
    for ds in datasets:
        inst_id = ds.instrument_meta.get("instrument_id", "unknown")
        if inst_id == reference_instrument_id:
            continue
        curve, diag = estimate_calibration_curve(ref, ds, max_shift_points=max_shift_points)
        curves[inst_id] = curve
        diagnostics[inst_id] = diag
    return curves, diagnostics


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
