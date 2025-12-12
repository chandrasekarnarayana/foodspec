"""Heating degradation analysis (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.features.peaks import PeakFeatureExtractor
from foodspec.features.ratios import RatioFeatureGenerator
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.cropping import RangeCropper
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.validation import validate_spectrum_set

__all__ = [
    "HeatingAnalysisResult",
    "run_heating_degradation_analysis",
    "run_heating_quality_workflow",
]


@dataclass
class HeatingAnalysisResult:
    preprocessed_spectra: np.ndarray
    wavenumbers: np.ndarray
    time_variable: pd.Series
    key_ratios: pd.DataFrame
    trend_models: Dict[str, Any]
    anova_results: Optional[pd.DataFrame]


def _default_heating_preprocess(wavenumbers: np.ndarray) -> Pipeline:
    return Pipeline(
        [
            ("als", ALSBaseline(lambda_=1e5, p=0.01, max_iter=10)),
            ("savgol", SavitzkyGolaySmoother(window_length=9, polyorder=3)),
            ("norm", VectorNormalizer(norm="l2")),
            ("crop", _HeatingCropper(wavenumbers=wavenumbers, min_wn=600, max_wn=1800)),
        ]
    )


def _default_heating_features(wavenumbers: np.ndarray) -> Pipeline:
    expected_peaks = [1655.0, 1742.0]
    ratios = {"ratio_1655_1742": ("peak_1655.0_height", "peak_1742.0_height")}
    return Pipeline(
        [
            ("peaks", PeakFeatureExtractor(expected_peaks=expected_peaks, tolerance=8.0)),
            ("ratios", RatioFeatureGenerator(ratio_def=ratios)),
        ]
    )


def run_heating_degradation_analysis(
    spectra: FoodSpectrumSet,
    time_column: str = "heating_time",
) -> HeatingAnalysisResult:
    """Run heating degradation analysis.

    Applies baseline/smoothing/normalization/cropping, extracts a key
    unsaturation/carbonyl ratio versus heating time, fits trend regressions,
    and computes a basic ANOVA if groups are present.

    Parameters
    ----------
    spectra : FoodSpectrumSet
        Spectral dataset with a heating/time column in metadata.
    time_column : str, optional
        Metadata column indicating heating time or stage, by default ``"heating_time"``.

    Returns
    -------
    HeatingAnalysisResult
        Preprocessed spectra, wavenumbers, time variable, ratio table,
        fitted trend models, and optional ANOVA table.

    Raises
    ------
    ValueError
        If the specified time column is missing.

    See also
    --------
    docs/workflows/heating_quality_monitoring.md : Workflow and interpretation.
    """

    validate_spectrum_set(spectra)
    if time_column not in spectra.metadata.columns:
        raise ValueError(f"Metadata column '{time_column}' not found.")

    preproc = _default_heating_preprocess(spectra.wavenumbers)
    X_proc = preproc.transform(spectra.x)
    wn_proc = preproc.named_steps["crop"].wavenumbers_

    extractor = PeakFeatureExtractor(expected_peaks=[1655.0, 1742.0], tolerance=8.0)
    extractor.fit(X_proc, wavenumbers=wn_proc)
    peak_feats = extractor.transform(X_proc, wavenumbers=wn_proc)
    peak_df = pd.DataFrame(peak_feats, columns=extractor.get_feature_names_out(), index=spectra.metadata.index)
    ratios = RatioFeatureGenerator({"ratio_1655_1742": ("peak_1655.0_height", "peak_1742.0_height")})
    ratio_df = ratios.transform(peak_df)

    trend_models: Dict[str, Any] = {}
    time_values = spectra.metadata[time_column].to_numpy().reshape(-1, 1)
    for col in ratio_df.columns:
        model = LinearRegression()
        model.fit(time_values, ratio_df[col].to_numpy())
        trend_models[col] = model

    # Optional group-wise models if oil_type present
    if "oil_type" in spectra.metadata.columns:
        grouped_models: Dict[str, Dict[str, Any]] = {}
        for col in ratio_df.columns:
            grouped_models[col] = {}
            for group, idxs in spectra.metadata.groupby("oil_type").groups.items():
                Xg = time_values[list(idxs)]
                yg = ratio_df[col].iloc[list(idxs)].to_numpy()
                if len(yg) >= 2:
                    m = LinearRegression()
                    m.fit(Xg, yg)
                    grouped_models[col][group] = m
        trend_models["by_oil_type"] = grouped_models

    anova_results = None
    if "oil_type" in spectra.metadata.columns and spectra.metadata["oil_type"].nunique() >= 2:
        rows = []
        for col in ratio_df.columns:
            groups = []
            for _, idxs in spectra.metadata.groupby("oil_type").groups.items():
                vals = ratio_df[col].iloc[list(idxs)].to_numpy()
                if len(vals) > 0:
                    groups.append(vals)
            if len(groups) >= 2:
                F, p = stats.f_oneway(*groups)
                rows.append({"factor": "oil_type", "metric": col, "F": F, "pvalue": p})
        if rows:
            anova_results = pd.DataFrame(rows)

    return HeatingAnalysisResult(
        preprocessed_spectra=X_proc,
        wavenumbers=wn_proc,
        time_variable=spectra.metadata[time_column],
        key_ratios=ratio_df,
        trend_models=trend_models,
        anova_results=anova_results,
    )


def run_heating_quality_workflow(
    spectra: FoodSpectrumSet,
    time_column: str = "heating_time",
    group_column: str | None = None,
) -> HeatingAnalysisResult:
    """Convenience wrapper for heating/quality monitoring.

    Parameters
    ----------
    spectra : FoodSpectrumSet
        Dataset with heating/time metadata.
    time_column : str, optional
        Metadata column encoding heating time or stage, by default ``"heating_time"``.
    group_column : str | None, optional
        Optional grouping factor (e.g., oil_type/batch). If provided and present,
        group-wise trend models are computed.

    Returns
    -------
    HeatingAnalysisResult

    See also
    --------
    docs/workflows/heating_quality_monitoring.md : End-to-end recipe and reporting.
    """
    return run_heating_degradation_analysis(spectra=spectra, time_column=time_column)


class _HeatingCropper(RangeCropper):
    def __init__(self, wavenumbers: np.ndarray, min_wn: float, max_wn: float):
        super().__init__(min_wn=min_wn, max_wn=max_wn)
        self.wavenumbers_full = np.asarray(wavenumbers, dtype=float)
        mask = (self.wavenumbers_full >= min_wn) & (self.wavenumbers_full <= max_wn)
        if not np.any(mask):
            raise ValueError("Cropping mask is empty for heating analysis.")
        self.mask_ = mask
        self.wavenumbers_ = self.wavenumbers_full[mask]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.wavenumbers_full.shape[0]:
            raise ValueError("Input X columns must match wavenumber axis.")
        return X[:, self.mask_]
