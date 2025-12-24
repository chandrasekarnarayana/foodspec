"""
Calibration transfer toolkit for spectroscopy.

This module provides tools for transferring calibration models between instruments:
- Direct Standardization (DS) and Piecewise Direct Standardization (PDS) v2
- Drift adaptation pipeline for periodic recalibration
- Transfer success metrics dashboard

**Key Assumptions:**
1. Source (reference) and target (slave) instruments measure the same samples
2. Standard/transfer samples span the calibration range
3. Spectral alignment is adequate (wavelength registration done separately)
4. Drift is gradual and can be modeled incrementally
5. Transfer samples are representative of production variability

**Typical Usage:**
    >>> from foodspec.calibration_transfer import direct_standardization
    >>> X_target_corrected = direct_standardization(X_source_std, X_target_std, X_target_prod)
    >>> # Or via exp.yml:
    >>> # calibration_transfer:
    >>> #   method: piecewise_ds
    >>> #   window_size: 11
    >>> #   source_dataset: reference_lib.csv
    >>> #   transfer_samples: transfer_set.csv
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Direct Standardization (DS)
# ──────────────────────────────────────────────────────────────────────────────


def direct_standardization(
    X_source_std: np.ndarray,
    X_target_std: np.ndarray,
    X_target_prod: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Direct Standardization (DS): transfer calibration from source to target instrument.
    
    Learns a linear transformation F such that X_target_std @ F ≈ X_source_std, then
    applies F to production spectra X_target_prod.
    
    **Assumptions:**
    - X_source_std and X_target_std are paired measurements of the same standards
    - Standards span the calibration range
    - Spectral dimensions match (same wavenumber axis)
    - Linear transformation is adequate (no severe nonlinearities)
    
    Reference: Wang et al. (1991), Multivariate Instrument Standardization
    
    Parameters
    ----------
    X_source_std : np.ndarray, shape (n_standards, n_wavenumbers)
        Source (reference) instrument spectra of standard samples.
    X_target_std : np.ndarray, shape (n_standards, n_wavenumbers)
        Target (slave) instrument spectra of the same standard samples.
    X_target_prod : np.ndarray, shape (n_prod, n_wavenumbers)
        Target instrument production spectra to be corrected.
    alpha : float, default=1.0
        Ridge regularization parameter (larger = more regularization).
        
    Returns
    -------
    X_target_corrected : np.ndarray, shape (n_prod, n_wavenumbers)
        Corrected target spectra (aligned to source domain).
    metrics : dict
        - 'reconstruction_rmse': RMSE on standard set
        - 'transformation_condition_number': condition number of F
        - 'n_standards': number of transfer standards used
    """
    n_standards, n_wavenumbers = X_source_std.shape
    assert X_target_std.shape == X_source_std.shape, "Source/target standard shapes must match."
    assert X_target_prod.shape[1] == n_wavenumbers, "Production spectra dimension mismatch."
    
    if n_standards < 10:
        warnings.warn(f"Only {n_standards} standards; DS may be unstable. Recommend ≥10.")
    
    # Fit transformation: F = (X_target_std.T @ X_target_std + alpha*I)^-1 @ X_target_std.T @ X_source_std
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_target_std, X_source_std)
    F = ridge.coef_.T  # (n_wavenumbers, n_wavenumbers)
    
    # Apply transformation to production spectra
    X_target_corrected = X_target_prod @ F
    
    # Compute reconstruction error on standards
    X_source_std_recon = X_target_std @ F
    recon_rmse = np.sqrt(mean_squared_error(X_source_std, X_source_std_recon))
    
    # Condition number (stability indicator)
    cond_number = np.linalg.cond(F)
    
    metrics = {
        "reconstruction_rmse": float(recon_rmse),
        "transformation_condition_number": float(cond_number),
        "n_standards": int(n_standards),
    }
    
    return X_target_corrected, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Piecewise Direct Standardization (PDS) v2
# ──────────────────────────────────────────────────────────────────────────────


def piecewise_direct_standardization(
    X_source_std: np.ndarray,
    X_target_std: np.ndarray,
    X_target_prod: np.ndarray,
    window_size: int = 11,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Piecewise Direct Standardization (PDS) v2 with robust regression.
    
    Learns local transformations per wavenumber window, reducing sensitivity to
    localized instrument differences.
    
    **Assumptions:**
    - Standards are paired and representative
    - Window size is odd and << n_wavenumbers
    - Local spectral neighborhoods are informative for each wavenumber
    - Robust regression (Ridge) prevents overfitting in small windows
    
    Reference: Bouveresse et al. (1996), Standardization of NIR spectra in diffuse reflectance mode
    
    Parameters
    ----------
    X_source_std : np.ndarray, shape (n_standards, n_wavenumbers)
        Source instrument standards.
    X_target_std : np.ndarray, shape (n_standards, n_wavenumbers)
        Target instrument standards.
    X_target_prod : np.ndarray, shape (n_prod, n_wavenumbers)
        Target production spectra.
    window_size : int, default=11
        Local window size (must be odd).
    alpha : float, default=1.0
        Ridge regularization.
        
    Returns
    -------
    X_target_corrected : np.ndarray, shape (n_prod, n_wavenumbers)
        Corrected spectra.
    metrics : dict
        - 'reconstruction_rmse': RMSE on standards
        - 'window_size': window size used
        - 'n_standards': number of standards
    """
    n_standards, n_wavenumbers = X_source_std.shape
    assert X_target_std.shape == X_source_std.shape
    assert X_target_prod.shape[1] == n_wavenumbers
    
    if window_size % 2 == 0:
        window_size += 1
        warnings.warn(f"window_size must be odd; adjusted to {window_size}.")
    
    half_window = window_size // 2
    
    X_target_corrected = np.empty_like(X_target_prod)
    
    # Build transformation for each wavenumber
    for j in range(n_wavenumbers):
        # Define local window
        j_start = max(0, j - half_window)
        j_end = min(n_wavenumbers, j + half_window + 1)
        
        # Local regression: X_target_std[:, window] -> X_source_std[:, j]
        X_local = X_target_std[:, j_start:j_end]
        y_local = X_source_std[:, j]
        
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_local, y_local)
        
        # Apply to production spectra
        X_prod_local = X_target_prod[:, j_start:j_end]
        X_target_corrected[:, j] = ridge.predict(X_prod_local)
    
    # Reconstruction error on standards
    X_source_std_recon = np.empty_like(X_source_std)
    for j in range(n_wavenumbers):
        j_start = max(0, j - half_window)
        j_end = min(n_wavenumbers, j + half_window + 1)
        X_local = X_target_std[:, j_start:j_end]
        y_local = X_source_std[:, j]
        ridge = Ridge(alpha=alpha, fit_intercept=False).fit(X_local, y_local)
        X_source_std_recon[:, j] = ridge.predict(X_local)
    
    recon_rmse = np.sqrt(mean_squared_error(X_source_std, X_source_std_recon))
    
    metrics = {
        "reconstruction_rmse": float(recon_rmse),
        "window_size": int(window_size),
        "n_standards": int(n_standards),
    }
    
    return X_target_corrected, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Drift Adaptation Pipeline
# ──────────────────────────────────────────────────────────────────────────────


def detect_drift(
    X_reference: np.ndarray,
    X_current: np.ndarray,
    threshold: float = 0.1,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect spectral drift by comparing current batch to reference.
    
    **Assumptions:**
    - X_reference and X_current are from the same matrix/sample type
    - Drift manifests as mean shift or variance change
    - Threshold is calibrated to acceptable drift magnitude
    
    Parameters
    ----------
    X_reference : np.ndarray, shape (n_ref, n_wavenumbers)
        Reference spectra (baseline).
    X_current : np.ndarray, shape (n_curr, n_wavenumbers)
        Current spectra to check for drift.
    threshold : float, default=0.1
        Drift threshold (normalized RMSE).
        
    Returns
    -------
    drift_detected : bool
        True if drift exceeds threshold.
    drift_metrics : dict
        - 'mean_shift': Euclidean distance between means
        - 'variance_ratio': ratio of variances
        - 'normalized_drift': drift magnitude / reference scale
    """
    mean_ref = X_reference.mean(axis=0)
    mean_curr = X_current.mean(axis=0)
    
    var_ref = X_reference.var(axis=0).mean()
    var_curr = X_current.var(axis=0).mean()
    
    mean_shift = np.linalg.norm(mean_curr - mean_ref)
    variance_ratio = var_curr / (var_ref + 1e-12)
    
    # Normalized drift score
    ref_scale = np.linalg.norm(mean_ref) + 1e-12
    normalized_drift = mean_shift / ref_scale
    
    drift_detected = normalized_drift > threshold
    
    drift_metrics = {
        "mean_shift": float(mean_shift),
        "variance_ratio": float(variance_ratio),
        "normalized_drift": float(normalized_drift),
        "threshold": float(threshold),
        "drift_detected": bool(drift_detected),
    }
    
    return drift_detected, drift_metrics


def adapt_calibration_incremental(
    X_reference: np.ndarray,
    X_new_standards: np.ndarray,
    weight_decay: float = 0.9,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Incrementally update reference set with new standards (exponential weighting).
    
    **Assumptions:**
    - X_new_standards are recent measurements of known standards
    - Drift is gradual (exponential decay appropriate)
    - Updated reference remains representative
    
    Parameters
    ----------
    X_reference : np.ndarray, shape (n_ref, n_wavenumbers)
        Current reference set.
    X_new_standards : np.ndarray, shape (n_new, n_wavenumbers)
        New standard measurements.
    weight_decay : float, default=0.9
        Weight for old reference (0 < weight_decay < 1).
        
    Returns
    -------
    X_reference_updated : np.ndarray, shape (n_ref + n_new, n_wavenumbers)
        Updated reference set.
    metrics : dict
        - 'reference_size_before': original reference size
        - 'reference_size_after': updated reference size
        - 'weight_decay': decay parameter used
    """
    # Weight old reference
    X_ref_weighted = weight_decay * X_reference
    
    # Append new standards
    X_reference_updated = np.vstack([X_ref_weighted, X_new_standards])
    
    metrics = {
        "reference_size_before": int(X_reference.shape[0]),
        "reference_size_after": int(X_reference_updated.shape[0]),
        "weight_decay": float(weight_decay),
    }
    
    return X_reference_updated, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Transfer Success Metrics Dashboard
# ──────────────────────────────────────────────────────────────────────────────


def compute_transfer_success_metrics(
    y_true: np.ndarray,
    y_pred_source: np.ndarray,
    y_pred_target_before: np.ndarray,
    y_pred_target_after: np.ndarray,
    X_target_prod: np.ndarray,
    leverage_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Comprehensive dashboard of transfer success metrics.
    
    **Assumptions:**
    - y_true are ground-truth labels/values for validation set
    - y_pred_source: predictions from source model (gold standard)
    - y_pred_target_before: predictions from target before transfer
    - y_pred_target_after: predictions from target after transfer
    - X_target_prod: production spectra for leverage/outlier detection
    
    Parameters
    ----------
    y_true : np.ndarray, shape (n_val,)
        Ground truth.
    y_pred_source : np.ndarray, shape (n_val,)
        Source model predictions.
    y_pred_target_before : np.ndarray, shape (n_val,)
        Target predictions before transfer.
    y_pred_target_after : np.ndarray, shape (n_val,)
        Target predictions after transfer.
    X_target_prod : np.ndarray, shape (n_prod, n_wavenumbers)
        Production spectra for leverage computation.
    leverage_threshold : float, default=0.3
        Leverage threshold for outlier flagging.
        
    Returns
    -------
    metrics : dict
        - Pre/post RMSE, R², MAE
        - Improvement ratios
        - Leverage/outlier counts
        - Residual statistics
    """
    # Prediction metrics
    rmse_source = np.sqrt(mean_squared_error(y_true, y_pred_source))
    rmse_before = np.sqrt(mean_squared_error(y_true, y_pred_target_before))
    rmse_after = np.sqrt(mean_squared_error(y_true, y_pred_target_after))
    
    r2_source = r2_score(y_true, y_pred_source)
    r2_before = r2_score(y_true, y_pred_target_before)
    r2_after = r2_score(y_true, y_pred_target_after)
    
    mae_source = mean_absolute_error(y_true, y_pred_source)
    mae_before = mean_absolute_error(y_true, y_pred_target_before)
    mae_after = mean_absolute_error(y_true, y_pred_target_after)
    
    # Improvement ratios
    rmse_improvement = (rmse_before - rmse_after) / (rmse_before + 1e-12)
    r2_improvement = r2_after - r2_before
    
    # Leverage computation (hat matrix diagonal)
    # Simplified: use Mahalanobis distance proxy
    X_mean = X_target_prod.mean(axis=0)
    X_centered = X_target_prod - X_mean
    cov = np.cov(X_centered, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    
    leverage = np.array([
        x @ cov_inv @ x for x in X_centered
    ])
    leverage_norm = leverage / leverage.mean()
    high_leverage_count = (leverage_norm > leverage_threshold).sum()
    
    # Residual analysis
    residuals_after = y_true - y_pred_target_after
    residual_mean = residuals_after.mean()
    residual_std = residuals_after.std()
    
    metrics = {
        "source_model": {
            "rmse": float(rmse_source),
            "r2": float(r2_source),
            "mae": float(mae_source),
        },
        "target_before_transfer": {
            "rmse": float(rmse_before),
            "r2": float(r2_before),
            "mae": float(mae_before),
        },
        "target_after_transfer": {
            "rmse": float(rmse_after),
            "r2": float(r2_after),
            "mae": float(mae_after),
        },
        "improvement": {
            "rmse_improvement_ratio": float(rmse_improvement),
            "r2_improvement_absolute": float(r2_improvement),
        },
        "leverage_outliers": {
            "high_leverage_count": int(high_leverage_count),
            "leverage_threshold": float(leverage_threshold),
            "leverage_rate": float(high_leverage_count / len(leverage_norm)),
        },
        "residuals": {
            "mean": float(residual_mean),
            "std": float(residual_std),
        },
    }
    
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# High-Level Workflow
# ──────────────────────────────────────────────────────────────────────────────


def calibration_transfer_workflow(
    X_source_std: np.ndarray,
    X_target_std: np.ndarray,
    X_target_prod: np.ndarray,
    method: Literal["ds", "pds"] = "ds",
    pds_window_size: int = 11,
    alpha: float = 1.0,
    y_val: Optional[np.ndarray] = None,
    y_pred_source_val: Optional[np.ndarray] = None,
    y_pred_target_val: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete calibration transfer workflow.
    
    **Assumptions:**
    - Standards are paired and representative
    - Validation set provided if transfer success metrics desired
    - Spectral alignment already performed
    
    Parameters
    ----------
    X_source_std : np.ndarray
        Source standards.
    X_target_std : np.ndarray
        Target standards.
    X_target_prod : np.ndarray
        Target production spectra.
    method : {'ds', 'pds'}, default='ds'
        Transfer method.
    pds_window_size : int, default=11
        PDS window size (ignored if method='ds').
    alpha : float, default=1.0
        Regularization parameter.
    y_val : np.ndarray, optional
        Validation labels (for metrics).
    y_pred_source_val : np.ndarray, optional
        Source predictions on validation set.
    y_pred_target_val : np.ndarray, optional
        Target predictions before transfer.
        
    Returns
    -------
    X_target_corrected : np.ndarray
        Corrected target spectra.
    all_metrics : dict
        Transfer metrics + success dashboard (if validation provided).
    """
    if method == "ds":
        X_target_corrected, transfer_metrics = direct_standardization(
            X_source_std, X_target_std, X_target_prod, alpha=alpha
        )
    elif method == "pds":
        X_target_corrected, transfer_metrics = piecewise_direct_standardization(
            X_source_std, X_target_std, X_target_prod, window_size=pds_window_size, alpha=alpha
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'ds' or 'pds'.")
    
    all_metrics = {"transfer": transfer_metrics}
    
    # Success metrics if validation data provided
    if y_val is not None and y_pred_source_val is not None and y_pred_target_val is not None:
        # Need to re-predict on corrected spectra (placeholder: assume improvement)
        y_pred_target_after_val = y_pred_target_val  # Placeholder; in real use, re-run model
        
        success_metrics = compute_transfer_success_metrics(
            y_val,
            y_pred_source_val,
            y_pred_target_val,
            y_pred_target_after_val,
            X_target_prod,
        )
        all_metrics["success_dashboard"] = success_metrics
    
    return X_target_corrected, all_metrics
