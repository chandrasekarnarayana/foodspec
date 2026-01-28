"""Microbial detection template."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.data_objects.spectra_set import FoodSpectrumSet

__all__ = ["MicrobialAnalysisResult", "run_microbial_detection_workflow"]


@dataclass
class MicrobialAnalysisResult:
    """Result container for microbial detection."""

    preprocessed_spectra: np.ndarray
    wavenumbers: np.ndarray
    cv_metrics: pd.DataFrame
    confusion_matrix: np.ndarray
    class_labels: list[str]


def run_microbial_detection_workflow(
    spectra: FoodSpectrumSet,
    label_column: str = "label",
    classifier_name: str = "rf",
    cv_splits: int = 5,
) -> MicrobialAnalysisResult:
    """Apply a generic authentication workflow to microbial spectra."""
    result = run_oil_authentication_workflow(
        spectra=spectra, label_column=label_column, classifier_name=classifier_name, cv_splits=cv_splits
    )
    preprocess = result.pipeline.named_steps.get("preprocess")
    if preprocess is not None:
        x_proc = preprocess.transform(spectra.x)
        wn_proc = preprocess.named_steps["crop"].wavenumbers_
    else:
        x_proc = spectra.x
        wn_proc = spectra.wavenumbers

    return MicrobialAnalysisResult(
        preprocessed_spectra=x_proc,
        wavenumbers=wn_proc,
        cv_metrics=result.cv_metrics,
        confusion_matrix=result.confusion_matrix,
        class_labels=result.class_labels,
    )
