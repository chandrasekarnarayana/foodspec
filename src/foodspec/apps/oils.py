"""Edible oil authentication workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.cropping import RangeCropper
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.features.peaks import PeakFeatureExtractor
from foodspec.features.ratios import RatioFeatureGenerator, compute_ratios
from foodspec.chemometrics.models import make_classifier
from foodspec.chemometrics.validation import compute_classification_metrics
from foodspec.validation import validate_spectrum_set

__all__ = [
    "OilAuthResult",
    "default_oil_preprocessing_pipeline",
    "default_oil_feature_pipeline",
    "run_oil_authentication_workflow",
    "run_oil_authentication_quickstart",
]


@dataclass
class OilAuthResult:
    """Results of the oil authentication workflow."""

    pipeline: Pipeline
    cv_metrics: pd.DataFrame
    confusion_matrix: np.ndarray
    class_labels: List[str]
    feature_importances: Optional[pd.Series]


class _RangeCropperTransformer(RangeCropper, TransformerMixin):
    """Pipeline-friendly wrapper over RangeCropper."""

    def __init__(self, wavenumbers: np.ndarray, min_wn: float, max_wn: float):
        self.wavenumbers_full = np.asarray(wavenumbers, dtype=float)
        super().__init__(min_wn=min_wn, max_wn=max_wn)
        mask = (self.wavenumbers_full >= self.min_wn) & (
            self.wavenumbers_full <= self.max_wn
        )
        if not np.any(mask):
            raise ValueError("Cropping mask is empty.")
        self.mask_ = mask
        self.wavenumbers_ = self.wavenumbers_full[mask]

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.wavenumbers_full.shape[0]:
            raise ValueError("Input X columns must match length of original wavenumbers.")
        return X[:, self.mask_]


class _PeakFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extract peak features with stored wavenumber axis."""

    def __init__(self, wavenumbers: np.ndarray, expected_peaks, tolerance: float = 5.0):
        self.wavenumbers = np.asarray(wavenumbers, dtype=float)
        self.expected_peaks = expected_peaks
        self.tolerance = tolerance
        self.extractor = PeakFeatureExtractor(
            expected_peaks=self.expected_peaks, tolerance=self.tolerance
        )
        self.feature_names_: List[str] = []

    def fit(self, X, y=None):
        self.extractor.fit(X, wavenumbers=self.wavenumbers)
        self.feature_names_ = list(self.extractor.get_feature_names_out())
        return self

    def transform(self, X):
        feats = self.extractor.transform(X, wavenumbers=self.wavenumbers)
        return pd.DataFrame(feats, columns=self.feature_names_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=str)


class _RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Compute ratio features and return DataFrame."""

    def __init__(self, ratio_def):
        self.ratio_def = ratio_def
        self.columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input to RatioFeatureTransformer must be a DataFrame.")
        self.columns_ = list(compute_ratios(X, self.ratio_def).columns)
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input to RatioFeatureTransformer must be a DataFrame.")
        df = compute_ratios(X, self.ratio_def)
        self.columns_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        if self.columns_ is None:
            return np.array([], dtype=str)
        return np.array(self.columns_, dtype=str)


class _DataFrameToArray(BaseEstimator, TransformerMixin):
    """Convert DataFrame features to numpy array while storing column names."""

    def __init__(self):
        self.columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame.")
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame.")
        self.columns_ = list(X.columns)
        return X.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if self.columns_ is None:
            return np.array([], dtype=str)
        return np.array(self.columns_, dtype=str)


def default_oil_preprocessing_pipeline(wavenumbers: np.ndarray) -> Pipeline:
    """Baseline, smoothing, normalization, and fingerprint cropping."""

    return Pipeline(
        steps=[
            ("als", ALSBaseline(lambda_=1e5, p=0.01, max_iter=10)),
            ("savgol", SavitzkyGolaySmoother(window_length=9, polyorder=3)),
            ("norm", VectorNormalizer(norm="l2")),
            ("crop", _RangeCropperTransformer(wavenumbers=wavenumbers, min_wn=600, max_wn=1800)),
        ]
    )


def default_oil_feature_pipeline(wavenumbers: np.ndarray) -> Pipeline:
    """Feature pipeline for oil peaks and ratios."""

    # Typical oil bands (approx): 1655 (C=C), 1742 (C=O), 1450 (CH2 bend)
    expected_peaks = [1655.0, 1742.0, 1450.0]
    ratio_def = {
        "ratio_1655_1742": ("peak_1655.0_height", "peak_1742.0_height"),
        "ratio_1450_1655": ("peak_1450.0_height", "peak_1655.0_height"),
    }

    return Pipeline(
        steps=[
            ("peaks", _PeakFeatureTransformer(wavenumbers=wavenumbers, expected_peaks=expected_peaks)),
            ("ratios", _RatioFeatureTransformer(ratio_def=ratio_def)),
            ("to_array", _DataFrameToArray()),
        ]
    )


def run_oil_authentication_workflow(
    spectra: FoodSpectrumSet,
    label_column: str = "oil_type",
    classifier_name: str = "rf",
    cv_splits: int = 5,
) -> OilAuthResult:
    """Run oil authentication pipeline with cross-validation.

    This wrapper applies a default preprocessing stack (baseline correction,
    Savitzky–Golay smoothing, L2 normalization, fingerprint crop), extracts
    peak/ratio features characteristic of edible oils, and trains a classifier.

    Parameters
    ----------
    spectra : FoodSpectrumSet
        Spectral library with an ``oil_type`` (or other) label column.
    label_column : str, optional
        Metadata column containing class labels, by default ``"oil_type"``.
    classifier_name : str, optional
        Classifier key supported by ``foodspec.chemometrics.models.make_classifier``
        (e.g., ``"rf"``, ``"svm_rbf"``, ``"logreg"``), by default ``"rf"``.
    cv_splits : int, optional
        Number of stratified folds for cross-validation, by default 5.

    Returns
    -------
    OilAuthResult
        Contains the fitted pipeline, cross-validation metrics, confusion matrix,
        class labels, and optional feature importances (when available).

    Raises
    ------
    ValueError
        If the specified label column is missing from metadata.

    See also
    --------
    docs/workflows/oil_authentication.md : Workflow recipe and interpretation.
    """

    validate_spectrum_set(spectra)
    if label_column not in spectra.metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")

    X = spectra.x
    y = spectra.metadata[label_column].to_numpy()
    classes = np.unique(y)

    preproc = default_oil_preprocessing_pipeline(spectra.wavenumbers)
    cropped_axis = preproc.named_steps["crop"].wavenumbers_
    feat_pipe = default_oil_feature_pipeline(cropped_axis)
    clf = make_classifier(classifier_name)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preproc),
            ("features", feat_pipe),
            ("clf", clf),
        ]
    )

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        pipeline.fit(X[train_idx], y[train_idx])
        preds = pipeline.predict(X[test_idx])
        y_true_fold = y[test_idx]
        metrics_df = compute_classification_metrics(y_true_fold, preds)
        metrics_row = metrics_df.iloc[0].to_dict()
        metrics_row["fold"] = fold
        fold_rows.append(metrics_row)

    metrics_df = pd.DataFrame(fold_rows)
    summary = metrics_df.drop(columns=["fold"]).agg(["mean", "std"])
    cv_metrics = pd.concat([metrics_df, summary.reset_index().rename(columns={"index": "fold"})])

    # Fit on full dataset
    pipeline.fit(X, y)
    preds_full = pipeline.predict(X)
    cm = confusion_matrix(y, preds_full, labels=classes)

    feature_importances = None
    clf_est = pipeline.named_steps["clf"]
    feature_names = pipeline.named_steps["features"].named_steps["to_array"].columns_
    if hasattr(clf_est, "feature_importances_"):
        feature_importances = pd.Series(
            clf_est.feature_importances_, index=feature_names, name="importance"
        )

    return OilAuthResult(
        pipeline=pipeline,
        cv_metrics=cv_metrics,
        confusion_matrix=cm,
        class_labels=classes.tolist(),
        feature_importances=feature_importances,
    )


def run_oil_authentication_quickstart(
    spectra: FoodSpectrumSet,
    label_column: str = "oil_type",
    classifier_name: str = "rf",
    cv_splits: int = 3,
) -> OilAuthResult:
    """Lightweight wrapper to run oil authentication with defaults.

    Parameters
    ----------
    spectra : FoodSpectrumSet
        Spectral library with an oil-type label column.
    label_column : str, optional
        Metadata column containing class labels, by default ``"oil_type"``.
    classifier_name : str, optional
        Classifier key (e.g., ``"rf"``, ``"svm_rbf"``), by default ``"rf"``.
    cv_splits : int, optional
        Number of CV folds, by default 3 (faster than the full workflow).

    Returns
    -------
    OilAuthResult

    Notes
    -----
    Intended for quick validation and tutorial examples; for publication runs,
    prefer ``run_oil_authentication_workflow`` with full CV.
    """

    return run_oil_authentication_workflow(
        spectra=spectra,
        label_column=label_column,
        classifier_name=classifier_name,
        cv_splits=cv_splits,
    )
