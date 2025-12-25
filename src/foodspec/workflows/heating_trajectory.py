"""
Heating and oxidation trajectory analysis for time-series spectroscopy.

This module provides tools for analyzing spectral changes over time:
- Time-series modeling on key oxidation/degradation indices
- Degradation stage classification
- Shelf-life estimation with confidence intervals
- Trajectory feature extraction

**Key Assumptions:**
1. Time column exists in metadata and is numeric (hours, days, or timestamps)
2. Samples are measured repeatedly over time (longitudinal data)
3. Degradation is monotonic or follows known patterns (linear, exponential, sigmoidal)
4. Sufficient time points per sample/group (≥5 recommended for regression)
5. No major batch effects confounding time trends

**Typical Usage:**
    >>> from foodspec import FoodSpec
    >>> fs = FoodSpec("heating_study.csv", modality="raman")
    >>> trajectory = fs.analyze_heating_trajectory(time_column="time_hours", indices=["pi", "tfc"])
    >>> shelf_life = trajectory["shelf_life_estimate"]
    >>> # Or via exp.yml:
    >>> # heating_trajectory:
    >>> #   time_column: time_hours
    >>> #   indices: [pi, tfc, oit]
    >>> #   classification: true
    >>> #   shelf_life_threshold: 2.0
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from foodspec.core.dataset import FoodSpectrumSet

# ──────────────────────────────────────────────────────────────────────────────
# Index Extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_oxidation_indices(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    indices: List[str] = ["pi", "tfc", "oit_proxy"],
) -> pd.DataFrame:
    """
    Extract common oxidation/degradation indices from spectra.

    **Assumptions:**
    - Wavenumbers are in cm⁻¹ (Raman shift)
    - Spectra are baseline-corrected and normalized
    - Peak regions are consistent with typical food matrices

    **Available Indices:**
    - 'pi': Peroxide Index (band ratio ~840 / ~1080 cm⁻¹)
    - 'tfc': Total Fatty Chain (intensity ~1440 cm⁻¹)
    - 'oit_proxy': Oxidative Induction Time proxy (ratio ~1660 / ~1440 cm⁻¹)
    - 'cc_stretch': C=C stretch intensity (~1660 cm⁻¹)
    - 'ch2_bend': CH₂ bending (~1440 cm⁻¹)

    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Spectral intensities.
    wavenumbers : np.ndarray, shape (n_wavenumbers,)
        Wavenumber axis in cm⁻¹.
    indices : list of str, default=['pi', 'tfc', 'oit_proxy']
        Indices to compute.

    Returns
    -------
    index_df : pd.DataFrame, shape (n_samples, len(indices))
        Computed indices.
    """
    n_samples = spectra.shape[0]
    index_values = {}

    def _intensity_at(wn_target: float, tol: float = 10.0) -> np.ndarray:
        """Extract intensity at target wavenumber ± tolerance."""
        idx = np.where((wavenumbers >= wn_target - tol) & (wavenumbers <= wn_target + tol))[0]
        if len(idx) == 0:
            warnings.warn(f"No wavenumbers found near {wn_target} cm⁻¹; returning zeros.")
            return np.zeros(n_samples)
        return spectra[:, idx].mean(axis=1)

    for index_name in indices:
        if index_name == "pi":
            # Peroxide Index: ratio of ~840 / ~1080
            i_840 = _intensity_at(840)
            i_1080 = _intensity_at(1080)
            index_values["pi"] = i_840 / (i_1080 + 1e-12)

        elif index_name == "tfc":
            # Total Fatty Chain: intensity at ~1440 (CH₂ bending)
            index_values["tfc"] = _intensity_at(1440)

        elif index_name == "oit_proxy":
            # OIT proxy: ratio ~1660 / ~1440
            i_1660 = _intensity_at(1660)
            i_1440 = _intensity_at(1440)
            index_values["oit_proxy"] = i_1660 / (i_1440 + 1e-12)

        elif index_name == "cc_stretch":
            # C=C stretch at ~1660
            index_values["cc_stretch"] = _intensity_at(1660)

        elif index_name == "ch2_bend":
            # CH₂ bending at ~1440
            index_values["ch2_bend"] = _intensity_at(1440)

        else:
            warnings.warn(f"Unknown index '{index_name}'; skipping.")

    index_df = pd.DataFrame(index_values)
    return index_df


# ──────────────────────────────────────────────────────────────────────────────
# Time-Series Modeling
# ──────────────────────────────────────────────────────────────────────────────


def fit_trajectory_model(
    time: np.ndarray,
    index_values: np.ndarray,
    model: Literal["linear", "exponential", "sigmoidal"] = "linear",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit a parametric model to index trajectory over time.

    **Assumptions:**
    - Time is sorted or will be sorted internally
    - No missing values in time or index_values
    - Sufficient data points (≥5) for reliable fitting

    Parameters
    ----------
    time : np.ndarray, shape (n_timepoints,)
        Time values (hours, days, etc.).
    index_values : np.ndarray, shape (n_timepoints,)
        Index measurements.
    model : {'linear', 'exponential', 'sigmoidal'}, default='linear'
        - 'linear': y = a + b*t
        - 'exponential': y = a * exp(b*t)
        - 'sigmoidal': y = L / (1 + exp(-k*(t - t0)))

    Returns
    -------
    fitted_values : np.ndarray, shape (n_timepoints,)
        Fitted trajectory.
    fit_metrics : dict
        - 'r_squared': coefficient of determination
        - 'rmse': root mean squared error
        - 'params': fitted parameters
        - 'trend_direction': 'increasing' or 'decreasing'
    """
    # Sort by time
    sort_idx = np.argsort(time)
    time_sorted = time[sort_idx]
    index_sorted = index_values[sort_idx]

    if model == "linear":
        # Linear fit: y = a + b*t
        params = np.polyfit(time_sorted, index_sorted, deg=1)
        fitted = np.polyval(params, time_sorted)
        fit_params = {"intercept": float(params[1]), "slope": float(params[0])}
        trend = "increasing" if params[0] > 0 else "decreasing"

    elif model == "exponential":
        # Exponential fit: y = a * exp(b*t)
        # Use log-linear regression for stability
        log_y = np.log(index_sorted + 1e-12)
        params_log = np.polyfit(time_sorted, log_y, deg=1)
        a = np.exp(params_log[1])
        b = params_log[0]
        fitted = a * np.exp(b * time_sorted)
        fit_params = {"a": float(a), "b": float(b)}
        trend = "increasing" if b > 0 else "decreasing"

    elif model == "sigmoidal":
        # Sigmoidal fit: y = L / (1 + exp(-k*(t - t0)))
        def sigmoid(t, L, k, t0):
            return L / (1 + np.exp(-k * (t - t0)))

        # Initial guess
        L_init = index_sorted.max()
        t0_init = time_sorted.mean()
        k_init = 0.1

        try:
            popt, _ = curve_fit(
                sigmoid,
                time_sorted,
                index_sorted,
                p0=[L_init, k_init, t0_init],
                maxfev=5000,
            )
            fitted = sigmoid(time_sorted, *popt)
            fit_params = {"L": float(popt[0]), "k": float(popt[1]), "t0": float(popt[2])}
            trend = "increasing" if popt[1] > 0 else "decreasing"
        except RuntimeError:
            warnings.warn("Sigmoidal fit failed; falling back to linear.")
            params = np.polyfit(time_sorted, index_sorted, deg=1)
            fitted = np.polyval(params, time_sorted)
            fit_params = {"intercept": float(params[1]), "slope": float(params[0])}
            trend = "increasing" if params[0] > 0 else "decreasing"

    # Compute fit metrics
    ss_res = ((index_sorted - fitted) ** 2).sum()
    ss_tot = ((index_sorted - index_sorted.mean()) ** 2).sum()
    r_squared = 1 - ss_res / (ss_tot + 1e-12)
    rmse = np.sqrt(ss_res / len(index_sorted))

    # Restore original order
    fitted_full = np.empty_like(index_values)
    fitted_full[sort_idx] = fitted

    fit_metrics = {
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "params": fit_params,
        "trend_direction": trend,
    }

    return fitted_full, fit_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Degradation Stage Classification
# ──────────────────────────────────────────────────────────────────────────────


def classify_degradation_stage(
    index_df: pd.DataFrame,
    stage_labels: np.ndarray,
    n_estimators: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train a degradation stage classifier from index features.

    **Assumptions:**
    - stage_labels are discrete categories (e.g., "fresh", "early", "advanced", "spoiled")
    - Sufficient samples per stage (≥10 recommended)
    - Indices are informative of degradation state

    Parameters
    ----------
    index_df : pd.DataFrame, shape (n_samples, n_indices)
        Extracted index features.
    stage_labels : np.ndarray, shape (n_samples,)
        Ground-truth degradation stage labels.
    n_estimators : int, default=100
        Number of trees in RandomForest.

    Returns
    -------
    predictions : np.ndarray, shape (n_samples,)
        Predicted stage labels.
    probabilities : np.ndarray, shape (n_samples, n_stages)
        Class probabilities.
    metrics : dict
        - 'cv_accuracy': cross-validation accuracy
        - 'feature_importance': importance scores per index
        - 'n_stages': number of unique stages
    """
    X = index_df.values
    y = stage_labels

    # Check class balance
    unique_stages, counts = np.unique(y, return_counts=True)
    n_stages = len(unique_stages)

    if n_stages < 2:
        raise ValueError("At least 2 degradation stages required for classification.")

    if counts.min() < 5:
        warnings.warn("Some stages have <5 samples; classification may be unreliable.")

    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
    )
    clf.fit(X, y)

    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)

    # Cross-validation accuracy
    cv_scores = cross_val_score(clf, X, y, cv=min(5, counts.min()), scoring="accuracy")
    cv_accuracy = cv_scores.mean()

    # Feature importance
    feature_importance = dict(zip(index_df.columns, clf.feature_importances_))

    metrics = {
        "cv_accuracy": float(cv_accuracy),
        "feature_importance": {k: float(v) for k, v in feature_importance.items()},
        "n_stages": int(n_stages),
        "stage_distribution": {str(stage): int(count) for stage, count in zip(unique_stages, counts)},
    }

    return predictions, probabilities, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Shelf-Life Estimation
# ──────────────────────────────────────────────────────────────────────────────


def _estimate_shelf_life(
    time: np.ndarray,
    index_values: np.ndarray,
    threshold: float,
    model: Literal["linear", "exponential"] = "linear",
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Estimate shelf life: time when index crosses threshold.

    **Assumptions:**
    - Index changes monotonically with time
    - Threshold represents spoilage/failure criterion
    - Extrapolation is valid within reasonable time range (warn if far)
    - Linear or exponential model captures trend adequately

    Parameters
    ----------
    time : np.ndarray, shape (n_timepoints,)
        Time values.
    index_values : np.ndarray, shape (n_timepoints,)
        Index measurements.
    threshold : float
        Threshold value for shelf-life criterion.
    model : {'linear', 'exponential'}, default='linear'
        Trajectory model.
    confidence_level : float, default=0.95
        Confidence level for interval estimation.

    Returns
    -------
    shelf_life_metrics : dict
        - 'shelf_life_estimate': estimated time to threshold
        - 'confidence_interval': (lower, upper) CI bounds
        - 'extrapolation_warning': bool, True if estimate is outside observed time range
        - 'fit_quality': R² of trajectory model
    """
    # Fit trajectory
    fitted, fit_metrics = fit_trajectory_model(time, index_values, model=model)
    r_squared = fit_metrics["r_squared"]
    params = fit_metrics["params"]

    # Solve for time when index = threshold
    if model == "linear":
        # y = a + b*t → t = (threshold - a) / b
        a = params["intercept"]
        b = params["slope"]
        if abs(b) < 1e-12:
            warnings.warn("Slope near zero; shelf-life estimation unreliable.")
            t_shelf = np.inf
        else:
            t_shelf = (threshold - a) / b

    elif model == "exponential":
        # y = a * exp(b*t) → t = ln(threshold / a) / b
        a = params["a"]
        b = params["b"]
        if abs(b) < 1e-12 or threshold / a <= 0:
            warnings.warn("Exponential model parameters invalid for threshold; returning inf.")
            t_shelf = np.inf
        else:
            t_shelf = np.log(threshold / a) / b

    # Check extrapolation
    t_min, t_max = time.min(), time.max()
    extrapolation_warning = not (t_min <= t_shelf <= t_max)

    # Confidence interval (bootstrap or parametric approximation)
    # Simplified: use residual standard error and t-distribution
    residuals = index_values - fitted
    residual_se = np.sqrt((residuals ** 2).sum() / (len(residuals) - 2))
    n = len(time)
    t_crit = t_dist.ppf((1 + confidence_level) / 2, df=n - 2)

    # Approximate CI for shelf life (assumes linear model)
    if model == "linear":
        # SE of predicted t: se_t ≈ residual_se / |b|
        se_t = residual_se / (abs(params["slope"]) + 1e-12)
        ci_lower = t_shelf - t_crit * se_t
        ci_upper = t_shelf + t_crit * se_t
    else:
        # For exponential, use wider interval (heuristic)
        se_t = residual_se / (abs(params["b"]) + 1e-12)
        ci_lower = t_shelf - t_crit * se_t
        ci_upper = t_shelf + t_crit * se_t

    shelf_life_metrics = {
        "shelf_life_estimate": float(t_shelf) if not np.isinf(t_shelf) else None,
        "confidence_interval": (float(ci_lower), float(ci_upper)),
        "extrapolation_warning": bool(extrapolation_warning),
        "fit_quality": float(r_squared),
        "model": model,
        "threshold": float(threshold),
    }

    return shelf_life_metrics


# ──────────────────────────────────────────────────────────────────────────────
# High-Level Workflow
# ──────────────────────────────────────────────────────────────────────────────


def analyze_heating_trajectory(
    dataset: FoodSpectrumSet,
    time_column: str,
    indices: List[str] = ["pi", "tfc", "oit_proxy"],
    classify_stages: bool = False,
    stage_column: Optional[str] = None,
    estimate_shelf_life: bool = False,
    shelf_life_threshold: Optional[float] = None,
    shelf_life_index: str = "pi",
) -> Dict[str, Any]:
    """
    Analyze heating/oxidation trajectory from time-series spectra.

    **Workflow:**
    1. Extract oxidation indices
    2. Fit trajectory models per index
    3. (Optional) Classify degradation stages
    4. (Optional) Estimate shelf life

    **Assumptions:**
    - time_column exists in metadata and is numeric
    - For stage classification: stage_column must be provided
    - For shelf-life estimation: shelf_life_threshold must be provided
    - Sufficient time points (≥5) for trajectory fitting

    Parameters
    ----------
    dataset : FoodSpectrumSet
        Input dataset with time-series spectra.
    time_column : str
        Metadata column with time values.
    indices : list of str, default=['pi', 'tfc', 'oit_proxy']
        Indices to extract and model.
    classify_stages : bool, default=False
        Whether to train degradation stage classifier.
    stage_column : str, optional
        Metadata column with degradation stage labels (required if classify_stages=True).
    estimate_shelf_life : bool, default=False
        Whether to estimate shelf life.
    shelf_life_threshold : float, optional
        Threshold for shelf-life criterion (required if estimate_shelf_life=True).
    shelf_life_index : str, default='pi'
        Index to use for shelf-life estimation.

    Returns
    -------
    results : dict
        - 'indices': DataFrame of extracted indices
        - 'trajectory_models': dict of fit metrics per index
        - 'stage_classification' (if enabled): classification metrics
        - 'shelf_life' (if enabled): shelf-life estimation metrics
    """
    # Validate inputs
    if time_column not in dataset.metadata.columns:
        raise ValueError(f"time_column '{time_column}' not found in metadata.")

    time = dataset.metadata[time_column].values
    if not np.issubdtype(time.dtype, np.number):
        raise ValueError(f"time_column '{time_column}' must be numeric.")

    if classify_stages and stage_column is None:
        raise ValueError("classify_stages=True requires stage_column to be specified.")

    if estimate_shelf_life and shelf_life_threshold is None:
        raise ValueError("estimate_shelf_life=True requires shelf_life_threshold to be specified.")

    results = {}

    # Step 1: Extract indices
    index_df = extract_oxidation_indices(dataset.x, dataset.wavenumbers, indices=indices)
    results["indices"] = index_df

    # Step 2: Fit trajectory models
    trajectory_models = {}
    for index_name in index_df.columns:
        index_vals = index_df[index_name].values
        fitted, fit_metrics = fit_trajectory_model(time, index_vals, model="linear")
        trajectory_models[index_name] = fit_metrics

    results["trajectory_models"] = trajectory_models

    # Step 3: Degradation stage classification
    if classify_stages:
        stage_labels = dataset.metadata[stage_column].values
        predictions, probabilities, class_metrics = classify_degradation_stage(index_df, stage_labels)
        results["stage_classification"] = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "metrics": class_metrics,
        }

    # Step 4: Shelf-life estimation
    if estimate_shelf_life:
        if shelf_life_index not in index_df.columns:
            raise ValueError(f"shelf_life_index '{shelf_life_index}' not found in extracted indices.")

        index_vals = index_df[shelf_life_index].values
        shelf_life_metrics = _estimate_shelf_life(
            time,
            index_vals,
            threshold=shelf_life_threshold,
            model="linear",
        )
        results["shelf_life"] = shelf_life_metrics

    return results
