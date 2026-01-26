"""Diagnostic & model evaluation methods.

Provides ROC/AUC analysis, threshold optimization, and statistical tests
for classification model evaluation.
"""

from .roc import (
    PerClassRocMetrics,
    RocDiagnosticsResult,
    ThresholdResult,
    compute_auc_ci_bootstrap,
    compute_binary_roc_diagnostics,
    compute_multiclass_roc_diagnostics,
    compute_roc_diagnostics,
)

__all__ = [
    "RocDiagnosticsResult",
    "PerClassRocMetrics",
    "ThresholdResult",
    "compute_roc_diagnostics",
    "compute_binary_roc_diagnostics",
    "compute_multiclass_roc_diagnostics",
    "compute_auc_ci_bootstrap",
]
