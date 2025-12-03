"""foodspec: Raman and FTIR spectroscopy toolkit for food science."""

__all__ = [
    "__version__",
    "FoodSpectrumSet",
    "HyperSpectralCube",
    "load_folder",
    "load_library",
    "create_library",
    "load_csv_spectra",
    "read_spectra",
    "detect_format",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_roc_curve",
    "compute_pr_curve",
    "run_anova",
    "run_ttest",
    "run_manova",
    "run_tukey_hsd",
    "run_kruskal_wallis",
    "run_mannwhitney_u",
    "run_wilcoxon_signed_rank",
    "run_friedman_test",
    "bootstrap_metric",
    "permutation_test_metric",
    "estimate_snr",
    "summarize_class_balance",
    "detect_outliers",
    "check_missing_metadata",
    "generate_synthetic_raman_spectrum",
    "generate_synthetic_ftir_spectrum",
]

# Single source of truth for the package version.
__version__ = "0.2.0"

from .core.dataset import FoodSpectrumSet
from .core.hyperspectral import HyperSpectralCube
from .io import (
    load_folder,
    load_library,
    create_library,
    load_csv_spectra,
    read_spectra,
    detect_format,
)
from .metrics import compute_classification_metrics, compute_regression_metrics, compute_roc_curve, compute_pr_curve
from .stats import (
    run_anova,
    run_ttest,
    run_manova,
    run_tukey_hsd,
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_wilcoxon_signed_rank,
    run_friedman_test,
    bootstrap_metric,
    permutation_test_metric,
)
from .utils.troubleshooting import estimate_snr, summarize_class_balance, detect_outliers, check_missing_metadata
from .synthetic import generate_synthetic_raman_spectrum, generate_synthetic_ftir_spectrum
