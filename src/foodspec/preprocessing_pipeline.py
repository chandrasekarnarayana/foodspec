"""
Preprocessing pipeline for RQ analysis.

Detects raw spectra vs precomputed peak tables, applies baseline/smoothing/normalization,
and extracts peak intensities for downstream RatioQualityEngine.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

from foodspec.rq import PeakDefinition
from foodspec.spectral_dataset import PreprocessingConfig  # reuse same config


def detect_input_mode(df: pd.DataFrame) -> str:
    """
    Heuristic detection of raw spectra vs peak table.
    - If any column starts with 'I_' => peak_table
    - Else if numeric wavenumber-like columns (>=3) => raw_spectra
    - Fallback: peak_table
    """
    if any(col.startswith("I_") for col in df.columns):
        return "peak_table"

    numeric_cols = []
    for col in df.columns:
        try:
            float(col)
            numeric_cols.append(col)
        except Exception:
            continue

    if len(numeric_cols) >= 3:
        return "raw_spectra"
    return "peak_table"


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline."""
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)


def extract_peaks_from_spectra(
    df: pd.DataFrame,
    peak_definitions: Iterable[PeakDefinition],
    wavenumber_cols: List[str],
) -> pd.DataFrame:
    """
    For each peak definition, find the max intensity within its window.
    Returns a new DataFrame with original metadata + I_<name> columns.
    """
    peaks = list(peak_definitions)
    wn = np.array([float(c) for c in wavenumber_cols])
    spectra = df[wavenumber_cols].to_numpy(dtype=float)
    meta_cols = [c for c in df.columns if c not in wavenumber_cols]
    out = df[meta_cols].copy()

    for peak in peaks:
        if peak.wavenumber is None:
            continue
        center = float(peak.wavenumber)
        low, high = peak.window if peak.window else (center - 10.0, center + 10.0)
        mask = (wn >= low) & (wn <= high)
        if not mask.any():
            out[peak.name] = np.nan
            continue
        region_indices = np.where(mask)[0]
        region_intensity = spectra[:, region_indices]
        max_idx = np.nanargmax(region_intensity, axis=1)
        peak_vals = region_intensity[np.arange(region_intensity.shape[0]), max_idx]
        out[peak.name] = peak_vals
    return out


def _normalize_spectra(X: np.ndarray, mode: str, wn: np.ndarray, ref_wn: float) -> np.ndarray:
    if mode == "none":
        return X
    Xn = X.copy()
    if mode == "vector":
        norms = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
        return Xn / norms
    if mode == "area":
        norms = np.sum(Xn, axis=1, keepdims=True) + 1e-12
        return Xn / norms
    if mode == "max":
        norms = np.max(Xn, axis=1, keepdims=True) + 1e-12
        return Xn / norms
    if mode == "reference":
        ref_idx = int(np.argmin(np.abs(wn - ref_wn)))
        norms = Xn[:, [ref_idx]] + 1e-12
        return Xn / norms
    return X


def run_full_preprocessing(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """
    If df is raw spectra: baseline -> smoothing -> normalization -> peak extraction.
    If df already has peaks, return as-is.
    """
    mode = detect_input_mode(df)
    if mode == "peak_table":
        return df

    peak_defs = getattr(config, "peak_definitions", []) or []
    # Separate wavenumber vs metadata
    wavenumber_cols = []
    meta_cols = []
    for col in df.columns:
        try:
            float(col)
            wavenumber_cols.append(col)
        except Exception:
            meta_cols.append(col)
    wn = np.array([float(c) for c in wavenumber_cols])
    spectra = df[wavenumber_cols].to_numpy(dtype=float)

    # Optional harmonization to provided grid
    if getattr(config, "align_to_common_grid", False):
        tgt = np.array(config.target_grid) if getattr(config, "target_grid", None) is not None else wn
        spectra = np.vstack([np.interp(tgt, wn, row) for row in spectra])
        wn = tgt

    # Baseline
    if getattr(config, "baseline_enabled", True):
        baseline_corrected = np.zeros_like(spectra)
        for i, row in enumerate(spectra):
            baseline_corrected[i, :] = row - baseline_als(row, lam=config.baseline_lambda, p=config.baseline_p)
        spectra = baseline_corrected

    # Smoothing
    if getattr(config, "smooth_enabled", True) and config.smooth_window > 2:
        spectra = savgol_filter(
            spectra,
            window_length=config.smooth_window,
            polyorder=config.smooth_polyorder,
            axis=1,
            mode="interp",
        )

    # Normalization
    spectra = _normalize_spectra(spectra, config.normalization, wn, config.reference_wavenumber)

    proc_df = pd.DataFrame(spectra, columns=wavenumber_cols, index=df.index)
    proc_df = pd.concat([df[meta_cols].reset_index(drop=True), proc_df.reset_index(drop=True)], axis=1)

    if peak_defs:
        proc_df = extract_peaks_from_spectra(proc_df, peak_defs, wavenumber_cols)
    return proc_df


__all__ = [
    "PreprocessingConfig",
    "detect_input_mode",
    "run_full_preprocessing",
    "extract_peaks_from_spectra",
]
