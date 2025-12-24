"""foodspec: Raman and FTIR spectroscopy toolkit for food science."""

__all__ = [
    "__version__",
    # Phase 1: Core unified entry point
    "FoodSpec",
    "Spectrum",
    "RunRecord",
    "OutputBundle",
    # Phase 0: Original exports
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
    "DatasetSpec",
    "ExperimentConfig",
    "ExperimentEngine",
    "diff_runs",
    "save_artifact",
    "load_artifact",
    "Predictor",
    # Moat modules: matrix correction, heating trajectory, calibration transfer, data governance
    "apply_matrix_correction",
    "analyze_heating_trajectory",
    "calibration_transfer_workflow",
    "direct_standardization",
    "piecewise_direct_standardization",
    # Data governance and dataset intelligence
    "summarize_dataset",
    "check_class_balance",
    "diagnose_imbalance",
    "compute_replicate_consistency",
    "assess_variability_sources",
    "detect_batch_label_correlation",
    "detect_replicate_leakage",
    "detect_leakage",
    "compute_readiness_score",
    # Plugin infrastructure
    "PluginManager",
    "install_plugin",
    "load_plugins",
]

# Single source of truth for the package version.
__version__ = "0.2.1"

# Phase 1: Core unified entry point and provenance
from .core.api import FoodSpec
from .core.spectrum import Spectrum
from .core.run_record import RunRecord
from .core.output_bundle import OutputBundle
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
from .repro import DatasetSpec, ExperimentConfig, ExperimentEngine, diff_runs
from .artifact import load_artifact, save_artifact, Predictor
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
from .matrix_correction import apply_matrix_correction
from .heating_trajectory import analyze_heating_trajectory
from .calibration_transfer import (
    calibration_transfer_workflow,
    direct_standardization,
    piecewise_direct_standardization,
)

# Data governance and dataset intelligence
from .core.summary import summarize_dataset
from .qc.dataset_qc import check_class_balance, diagnose_imbalance
from .qc.replicates import compute_replicate_consistency, assess_variability_sources
from .qc.leakage import detect_batch_label_correlation, detect_replicate_leakage, detect_leakage
from .qc.readiness import compute_readiness_score
from .plugin import PluginManager, install_plugin
from .plugins import load_plugins
