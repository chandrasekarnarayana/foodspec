from __future__ import annotations
"""
Matrix correction for food spectroscopy: background subtraction, robust scaling, domain adaptation.

This module provides tools to correct for matrix effects (e.g., chips vs. pure oil) in spectroscopy:
- Background subtraction strategies (air, dark, foil references, adaptive baseline)
- Robust per-matrix scaling (median/MAD, Huber scaling)
- Domain adaptation presets for common matrix archetypes
- Matrix effect magnitude metric for QC reporting

**Key Assumptions:**
1. Background reference spectra are measured under identical conditions (instrument, temp)
2. Matrix types are known or can be inferred from metadata columns
3. Domain adaptation requires sufficient samples from both source and target matrices
4. Spectral ranges should align before correction (no wavelength drift)

**Typical Usage:**
    >>> from foodspec import FoodSpec
    >>> fs = FoodSpec("data.csv", modality="raman")
    >>> fs.apply_matrix_correction(method="background_air", matrix_column="matrix_type")
    >>> # Or via exp.yml:
    >>> # matrix_correction:
    >>> #   method: background_air
    >>> #   scaling: robust_mad
    >>> #   matrix_column: matrix_type
"""


import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler

from foodspec.core.dataset import FoodSpectrumSet

# ──────────────────────────────────────────────────────────────────────────────
# Background Subtraction
# ──────────────────────────────────────────────────────────────────────────────


def subtract_background_reference(
    spectra: np.ndarray,
    reference: np.ndarray,
    method: Literal["direct", "scaled", "adaptive"] = "scaled",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Subtract a background reference spectrum from sample spectra.

    **Assumptions:**
    - Reference spectrum is measured under identical conditions
    - Reference is a 1D array (single spectrum) or 2D array (multiple references to average)
    - Spectra and reference share the same wavenumber axis

    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Sample spectra to correct.
    reference : np.ndarray, shape (n_wavenumbers,) or (n_refs, n_wavenumbers)
        Background reference spectrum/spectra.
    method : {'direct', 'scaled', 'adaptive'}, default='scaled'
        - 'direct': subtract reference as-is
        - 'scaled': scale reference intensity to match sample baseline before subtracting
        - 'adaptive': per-sample scaling factor based on low-frequency match

    Returns
    -------
    corrected : np.ndarray, shape (n_samples, n_wavenumbers)
        Background-corrected spectra.
    metrics : dict
        - 'reference_intensity_mean': mean reference intensity
        - 'scaling_factors': array of per-sample scaling factors (if adaptive)
        - 'residual_baseline_shift': mean baseline shift after correction
    """
    if reference.ndim == 2:
        # Average multiple reference spectra
        reference = reference.mean(axis=0)

    n_samples, n_wavenumbers = spectra.shape
    assert reference.shape[0] == n_wavenumbers, "Reference shape mismatch"

    corrected = np.empty_like(spectra)
    scaling_factors = np.ones(n_samples)

    if method == "direct":
        # Simple subtraction
        corrected = spectra - reference[np.newaxis, :]

    elif method == "scaled":
        # Scale reference to match median baseline of dataset
        ref_median = np.median(reference)
        dataset_baseline = np.median(spectra, axis=1).mean()
        scale = dataset_baseline / (ref_median + 1e-12)
        corrected = spectra - scale * reference[np.newaxis, :]
        scaling_factors[:] = scale

    elif method == "adaptive":
        # Per-sample scaling: match reference to sample baseline in low-frequency region
        # Use first 10% and last 10% of spectrum as baseline proxy
        n_edge = max(1, n_wavenumbers // 10)
        for i in range(n_samples):
            sample_baseline = np.concatenate([spectra[i, :n_edge], spectra[i, -n_edge:]]).mean()
            ref_baseline = np.concatenate([reference[:n_edge], reference[-n_edge:]]).mean()
            scale = sample_baseline / (ref_baseline + 1e-12)
            corrected[i] = spectra[i] - scale * reference
            scaling_factors[i] = scale

    # Compute metrics
    residual_baseline = np.median(corrected, axis=1).mean()

    metrics = {
        "reference_intensity_mean": float(reference.mean()),
        "scaling_factors": scaling_factors.tolist() if method == "adaptive" else [float(scaling_factors[0])],
        "residual_baseline_shift": float(residual_baseline),
    }

    return corrected, metrics


def adaptive_baseline_correction(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Asymmetric least squares (ALS) baseline correction.

    Fits a smooth baseline to each spectrum by penalizing curvature and asymmetrically
    weighting residuals (favoring under-fitting to avoid removing peaks).

    **Assumptions:**
    - Baseline is smooth and monotonic or slowly varying
    - Peaks are positive (absorbance/intensity mode)
    - No negative-going features that should be preserved

    Reference: Eilers & Boelens (2005), Baseline Correction with Asymmetric Least Squares Smoothing

    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Spectra to correct.
    lam : float, default=1e5
        Smoothness parameter (larger = smoother baseline).
    p : float, default=0.01
        Asymmetry parameter (0 < p < 1, smaller favors under-fitting).
    niter : int, default=10
        Number of iterations.

    Returns
    -------
    corrected : np.ndarray, shape (n_samples, n_wavenumbers)
        Baseline-corrected spectra.
    metrics : dict
        - 'baseline_intensity_mean': mean baseline intensity
        - 'correction_magnitude': RMS of baseline
    """
    n_samples, n_wavenumbers = spectra.shape
    corrected = np.empty_like(spectra)
    baselines = np.empty_like(spectra)

    # Construct difference matrix for smoothness penalty
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n_wavenumbers - 2, n_wavenumbers))
    D = D.tocsc()

    for i in range(n_samples):
        y = spectra[i]
        w = np.ones(n_wavenumbers)

        for _ in range(niter):
            W = sparse.diags(w, 0, shape=(n_wavenumbers, n_wavenumbers))
            Z = W + lam * D.T @ D
            z = spsolve(Z, w * y)

            # Update weights: penalize positive residuals more
            w = p * (y > z) + (1 - p) * (y <= z)

        baselines[i] = z
        corrected[i] = y - z

    baseline_mean = baselines.mean()
    correction_magnitude = np.sqrt((baselines**2).mean())

    metrics = {
        "baseline_intensity_mean": float(baseline_mean),
        "correction_magnitude": float(correction_magnitude),
    }

    return corrected, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Robust Per-Matrix Scaling
# ──────────────────────────────────────────────────────────────────────────────


def robust_scale_per_matrix(
    spectra: np.ndarray,
    matrix_labels: np.ndarray,
    method: Literal["median_mad", "huber", "mcd"] = "median_mad",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Robust scaling per matrix type to normalize intensity distributions.

    **Assumptions:**
    - Matrix labels are provided and accurate
    - Each matrix group has sufficient samples (≥5 recommended)
    - Scaling is applied within-matrix to preserve relative intensities

    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Spectra to scale.
    matrix_labels : np.ndarray, shape (n_samples,)
        Matrix type labels (e.g., "chips", "pure_oil", "emulsion").
    method : {'median_mad', 'huber', 'mcd'}, default='median_mad'
        - 'median_mad': median centering + MAD scaling
        - 'huber': RobustScaler (sklearn) with quantile-based centering
        - 'mcd': Minimum Covariance Determinant robust estimator

    Returns
    -------
    scaled : np.ndarray, shape (n_samples, n_wavenumbers)
        Scaled spectra.
    metrics : dict
        - 'matrix_scaling_stats': per-matrix scaling parameters
        - 'matrix_counts': sample count per matrix
    """
    unique_matrices = np.unique(matrix_labels)
    scaled = np.empty_like(spectra)
    scaling_stats = {}
    matrix_counts = {}

    for matrix in unique_matrices:
        mask = matrix_labels == matrix
        n_matrix = mask.sum()
        matrix_counts[str(matrix)] = int(n_matrix)

        if n_matrix < 3:
            warnings.warn(f"Matrix '{matrix}' has only {n_matrix} samples; scaling may be unreliable.")

        X = spectra[mask]

        if method == "median_mad":
            # Median centering + MAD scaling
            median = np.median(X, axis=0)
            mad = np.median(np.abs(X - median), axis=0)
            mad = np.where(mad == 0, 1.0, mad)  # Avoid division by zero
            X_scaled = (X - median) / mad
            scaling_stats[str(matrix)] = {
                "median": median.mean(),
                "mad": mad.mean(),
            }

        elif method == "huber":
            # RobustScaler (quantile-based)
            scaler = RobustScaler(quantile_range=(25.0, 75.0))
            X_scaled = scaler.fit_transform(X)
            scaling_stats[str(matrix)] = {
                "center": float(scaler.center_.mean()),
                "scale": float(scaler.scale_.mean()),
            }

        elif method == "mcd":
            # Minimum Covariance Determinant (multivariate robust)
            if n_matrix < 10:
                warnings.warn(
                    f"MCD scaling requires ≥10 samples; {n_matrix} found for '{matrix}'. Falling back to median_mad."
                )
                median = np.median(X, axis=0)
                mad = np.median(np.abs(X - median), axis=0)
                mad = np.where(mad == 0, 1.0, mad)
                X_scaled = (X - median) / mad
                scaling_stats[str(matrix)] = {"median": median.mean(), "mad": mad.mean()}
            else:
                mcd = MinCovDet(random_state=42).fit(X)
                center = mcd.location_
                scale = np.sqrt(np.diag(mcd.covariance_))
                scale = np.where(scale == 0, 1.0, scale)
                X_scaled = (X - center) / scale
                scaling_stats[str(matrix)] = {
                    "center": float(center.mean()),
                    "scale": float(scale.mean()),
                }

        scaled[mask] = X_scaled

    metrics = {
        "matrix_scaling_stats": scaling_stats,
        "matrix_counts": matrix_counts,
    }

    return scaled, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Domain Adaptation Presets
# ──────────────────────────────────────────────────────────────────────────────


def domain_adapt_subspace_alignment(
    source_spectra: np.ndarray,
    target_spectra: np.ndarray,
    n_components: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Subspace alignment for domain adaptation between matrix types.

    Aligns principal subspaces of source and target domains to reduce distribution shift.

    **Assumptions:**
    - Source and target share the same wavenumber axis
    - Source has labels; target may be unlabeled
    - n_components << min(n_samples_source, n_samples_target)
    - Linear subspace structure approximates domain differences

    Reference: Fernando et al. (2013), Unsupervised Visual Domain Adaptation Using Subspace Alignment

    Parameters
    ----------
    source_spectra : np.ndarray, shape (n_source, n_wavenumbers)
        Source domain spectra (e.g., pure oil reference library).
    target_spectra : np.ndarray, shape (n_target, n_wavenumbers)
        Target domain spectra (e.g., chips to be corrected).
    n_components : int, default=10
        Number of principal components for subspace.

    Returns
    -------
    target_aligned : np.ndarray, shape (n_target, n_wavenumbers)
        Target spectra aligned to source domain.
    metrics : dict
        - 'alignment_shift_magnitude': Frobenius norm of alignment transformation
        - 'explained_variance_source': variance captured by source PCA
        - 'explained_variance_target': variance captured by target PCA
    """
    # Center data
    source_mean = source_spectra.mean(axis=0)
    target_mean = target_spectra.mean(axis=0)
    X_s = source_spectra - source_mean
    X_t = target_spectra - target_mean

    # Ensure n_components doesn't exceed sample count
    n_components = min(n_components, X_s.shape[0], X_t.shape[0], X_s.shape[1])

    # PCA on source
    U_s, s_s, Vt_s = np.linalg.svd(X_s, full_matrices=False)
    U_s = U_s[:, :n_components]
    Vt_s = Vt_s[:n_components, :]
    explained_var_s = (s_s[:n_components] ** 2).sum() / (s_s**2).sum()

    # PCA on target
    U_t, s_t, Vt_t = np.linalg.svd(X_t, full_matrices=False)
    U_t = U_t[:, :n_components]
    Vt_t = Vt_t[:n_components, :]
    explained_var_t = (s_t[:n_components] ** 2).sum() / (s_t**2).sum()

    # Compute alignment matrix: M = Vt_s @ Vt_t.T
    M = Vt_s @ Vt_t.T

    # Transform target to source subspace
    X_t_proj = X_t @ Vt_t.T  # Project target onto target subspace
    X_t_aligned = X_t_proj @ M @ Vt_s  # Rotate and project back
    target_aligned = X_t_aligned + source_mean  # Re-center to source domain

    alignment_shift = np.linalg.norm(M - np.eye(n_components), ord="fro")

    metrics = {
        "alignment_shift_magnitude": float(alignment_shift),
        "explained_variance_source": float(explained_var_s),
        "explained_variance_target": float(explained_var_t),
        "n_components": n_components,
    }

    return target_aligned, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Matrix Effect Magnitude Metric
# ──────────────────────────────────────────────────────────────────────────────


def compute_matrix_effect_magnitude(
    spectra_before: np.ndarray,
    spectra_after: np.ndarray,
    matrix_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute matrix effect magnitude before/after correction.

    Quantifies how much matrix bias was removed by correction pipeline.

    **Assumptions:**
    - spectra_before and spectra_after are aligned (same samples, same order)
    - If matrix_labels provided, per-matrix stats are computed

    Parameters
    ----------
    spectra_before : np.ndarray, shape (n_samples, n_wavenumbers)
        Spectra before correction.
    spectra_after : np.ndarray, shape (n_samples, n_wavenumbers)
        Spectra after correction.
    matrix_labels : np.ndarray, optional, shape (n_samples,)
        Matrix type labels for per-matrix reporting.

    Returns
    -------
    metrics : dict
        - 'total_correction_magnitude': RMS of (before - after)
        - 'baseline_shift_before': mean baseline before correction
        - 'baseline_shift_after': mean baseline after correction
        - 'per_matrix_correction' (if labels provided): correction magnitude per matrix
    """
    delta = spectra_before - spectra_after
    correction_magnitude = np.sqrt((delta**2).mean())

    baseline_before = np.median(spectra_before, axis=1).mean()
    baseline_after = np.median(spectra_after, axis=1).mean()

    metrics = {
        "total_correction_magnitude": float(correction_magnitude),
        "baseline_shift_before": float(baseline_before),
        "baseline_shift_after": float(baseline_after),
    }

    if matrix_labels is not None:
        per_matrix = {}
        for matrix in np.unique(matrix_labels):
            mask = matrix_labels == matrix
            delta_m = delta[mask]
            mag_m = np.sqrt((delta_m**2).mean())
            per_matrix[str(matrix)] = float(mag_m)
        metrics["per_matrix_correction"] = per_matrix

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# High-Level Workflow
# ──────────────────────────────────────────────────────────────────────────────


def apply_matrix_correction(
    dataset: FoodSpectrumSet,
    method: Literal["background_air", "background_dark", "adaptive_baseline", "none"] = "adaptive_baseline",
    scaling: Literal["median_mad", "huber", "mcd", "none"] = "median_mad",
    domain_adapt: bool = False,
    matrix_column: Optional[str] = None,
    reference_spectra: Optional[np.ndarray] = None,
) -> Tuple[FoodSpectrumSet, Dict[str, Any]]:
    """
    Apply full matrix correction pipeline to a dataset.

    **Workflow:**
    1. Background subtraction (if reference provided or adaptive baseline)
    2. Robust per-matrix scaling (if matrix_column provided)
    3. Domain adaptation (if enabled and matrix_column provided)
    4. Compute matrix effect magnitude metrics

    **Assumptions:**
    - Dataset has consistent wavenumber axis
    - If using background subtraction, reference_spectra must be provided
    - If using scaling or domain adaptation, matrix_column must exist in metadata
    - Domain adaptation requires ≥2 matrix types with ≥10 samples each

    Parameters
    ----------
    dataset : FoodSpectrumSet
        Input dataset.
    method : str, default='adaptive_baseline'
        Background subtraction method.
    scaling : str, default='median_mad'
        Robust scaling method.
    domain_adapt : bool, default=False
        Whether to apply domain adaptation.
    matrix_column : str, optional
        Metadata column with matrix type labels.
    reference_spectra : np.ndarray, optional
        Background reference spectra (for background_air/dark methods).

    Returns
    -------
    corrected_dataset : FoodSpectrumSet
        Corrected dataset.
    metrics : dict
        Combined metrics from all correction steps.
    """
    spectra_original = dataset.x.copy()
    spectra = spectra_original.copy()
    all_metrics = {}

    # Step 1: Background subtraction
    if method in ["background_air", "background_dark"]:
        if reference_spectra is None:
            raise ValueError(f"method='{method}' requires reference_spectra to be provided.")
        spectra, bg_metrics = subtract_background_reference(spectra, reference_spectra, method="adaptive")
        all_metrics["background_subtraction"] = bg_metrics

    elif method == "adaptive_baseline":
        spectra, baseline_metrics = adaptive_baseline_correction(spectra)
        all_metrics["adaptive_baseline"] = baseline_metrics

    # Step 2: Robust scaling per matrix
    if scaling != "none":
        if matrix_column is None:
            raise ValueError(f"scaling='{scaling}' requires matrix_column to be specified.")
        if matrix_column not in dataset.metadata.columns:
            raise ValueError(f"matrix_column '{matrix_column}' not found in dataset metadata.")

        matrix_labels = dataset.metadata[matrix_column].values
        spectra, scale_metrics = robust_scale_per_matrix(spectra, matrix_labels, method=scaling)
        all_metrics["robust_scaling"] = scale_metrics

    # Step 3: Domain adaptation
    if domain_adapt:
        if matrix_column is None:
            raise ValueError("domain_adapt requires matrix_column to be specified.")

        matrix_labels = dataset.metadata[matrix_column].values
        unique_matrices = np.unique(matrix_labels)

        if len(unique_matrices) < 2:
            warnings.warn("domain_adapt requires ≥2 matrix types; skipping.")
        else:
            # Use first matrix as source, rest as targets
            source_matrix = unique_matrices[0]
            source_mask = matrix_labels == source_matrix
            source_spectra = spectra[source_mask]

            for target_matrix in unique_matrices[1:]:
                target_mask = matrix_labels == target_matrix
                if target_mask.sum() < 10:
                    warnings.warn(f"Skipping domain adaptation for '{target_matrix}': <10 samples.")
                    continue

                target_spectra = spectra[target_mask]
                target_aligned, adapt_metrics = domain_adapt_subspace_alignment(source_spectra, target_spectra)
                spectra[target_mask] = target_aligned
                all_metrics[f"domain_adaptation_{target_matrix}"] = adapt_metrics

    # Step 4: Compute matrix effect magnitude
    matrix_labels_for_metric = None
    if matrix_column is not None and matrix_column in dataset.metadata.columns:
        matrix_labels_for_metric = dataset.metadata[matrix_column].values

    effect_metrics = compute_matrix_effect_magnitude(
        spectra_original,
        spectra,
        matrix_labels=matrix_labels_for_metric,
    )
    all_metrics["matrix_effect_magnitude"] = effect_metrics

    # Build corrected dataset
    corrected_dataset = FoodSpectrumSet(
        x=spectra,
        wavenumbers=dataset.wavenumbers,
        metadata=dataset.metadata.copy(),
        modality=dataset.modality,
    )

    return corrected_dataset, all_metrics
