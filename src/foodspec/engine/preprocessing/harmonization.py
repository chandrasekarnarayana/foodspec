"""Instrument harmonization helpers (engine namespace)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from foodspec.preprocess.calibration_transfer import direct_standardization, piecewise_direct_standardization
from foodspec.preprocess.matrix_correction import domain_adapt_subspace_alignment


def apply_direct_standardization(
    X_source_std: np.ndarray,
    X_target_std: np.ndarray,
    X_target_prod: np.ndarray,
    *,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply direct standardization (DS) for instrument transfer."""
    return direct_standardization(X_source_std, X_target_std, X_target_prod, alpha=alpha)


def apply_piecewise_direct_standardization(
    X_source_std: np.ndarray,
    X_target_std: np.ndarray,
    X_target_prod: np.ndarray,
    *,
    window_size: int = 11,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply piecewise direct standardization (PDS) for instrument transfer."""
    return piecewise_direct_standardization(
        X_source_std,
        X_target_std,
        X_target_prod,
        window_size=window_size,
        alpha=alpha,
    )


def apply_subspace_alignment(
    source_spectra: np.ndarray,
    target_spectra: np.ndarray,
    *,
    n_components: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply subspace alignment for domain adaptation between instruments."""
    return domain_adapt_subspace_alignment(source_spectra, target_spectra, n_components=n_components)


__all__ = [
    "apply_direct_standardization",
    "apply_piecewise_direct_standardization",
    "apply_subspace_alignment",
]
