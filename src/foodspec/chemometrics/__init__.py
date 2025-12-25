"""Chemometric model helpers."""

from .mixture import mcr_als, nnls_mixture, run_mixture_analysis_workflow
from .models import (
    make_classifier,
    make_mlp_regressor,
    make_pls_da,
    make_pls_regression,
    make_simca,
    SIMCAClassifier,
    make_regressor,
    make_one_class_scanner,
)
from .pca import run_pca
from .validation import (
    compute_classification_metrics,
    compute_regression_metrics,
    cross_validate_pipeline,
    permutation_test_score_wrapper,
    compute_explained_variance,
)
from .vip import (
    calculate_vip,
    calculate_vip_da,
    interpret_vip,
)

__all__ = [
    "make_classifier",
    "make_pls_da",
    "make_pls_regression",
    "make_mlp_regressor",
    "make_simca",
    "SIMCAClassifier",
    "make_regressor",
    "make_one_class_scanner",
    "run_pca",
    "nnls_mixture",
    "mcr_als",
    "run_mixture_analysis_workflow",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "cross_validate_pipeline",
    "permutation_test_score_wrapper",
    "compute_explained_variance",
    "calculate_vip",
    "calculate_vip_da",
    "interpret_vip",
]
