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
    "LOGO_BASE64",
    "get_logo_bytes",
    "get_logo_base64",
    "save_logo",
    "PeakDefinition",
    "RatioDefinition",
    "RQConfig",
    "RatioQualityEngine",
    "RatioQualityResult",
]

# Single source of truth for the package version.
__version__ = "0.2.1"

from .core.dataset import FoodSpectrumSet
from .core.hyperspectral import HyperSpectralCube
from .io import (
    create_library,
    detect_format,
    load_csv_spectra,
    load_folder,
    load_library,
    read_spectra,
)
from .logo import LOGO_BASE64, get_logo_base64, get_logo_bytes, save_logo
from .metrics import (
    compute_classification_metrics,
    compute_pr_curve,
    compute_regression_metrics,
    compute_roc_curve,
)
from .rq import (
    PeakDefinition,
    RatioDefinition,
    RatioQualityEngine,
    RatioQualityResult,
    RQConfig,
)
from .stats import (
    bootstrap_metric,
    permutation_test_metric,
    run_anova,
    run_friedman_test,
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_manova,
    run_ttest,
    run_tukey_hsd,
    run_wilcoxon_signed_rank,
)
from .synthetic import (
    generate_synthetic_ftir_spectrum,
    generate_synthetic_raman_spectrum,
)
from .utils.troubleshooting import (
    check_missing_metadata,
    detect_outliers,
    estimate_snr,
    summarize_class_balance,
)
