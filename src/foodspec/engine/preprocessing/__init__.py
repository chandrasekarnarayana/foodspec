"""Preprocessing steps exposed via the engine namespace."""
from __future__ import annotations

from .baseline import baseline_als, baseline_polynomial, baseline_rubberband
from .engine import (
    AlignmentStep,
    AutoPreprocess,
    AutoPreprocessResult,
    BaselineStep,
    DerivativeStep,
    NormalizationStep,
    PreprocessPipeline,
    ResampleStep,
    SmoothingStep,
    Step,
)
from .harmonization import (
    apply_direct_standardization,
    apply_piecewise_direct_standardization,
    apply_subspace_alignment,
)
from .normalization import normalize_reference, normalize_vector
from .smoothing import smooth_savgol

__all__ = [
    "baseline_als",
    "baseline_polynomial",
    "baseline_rubberband",
    "normalize_reference",
    "normalize_vector",
    "smooth_savgol",
    "apply_direct_standardization",
    "apply_piecewise_direct_standardization",
    "apply_subspace_alignment",
    "AlignmentStep",
    "AutoPreprocess",
    "AutoPreprocessResult",
    "BaselineStep",
    "DerivativeStep",
    "NormalizationStep",
    "PreprocessPipeline",
    "ResampleStep",
    "SmoothingStep",
    "Step",
]
