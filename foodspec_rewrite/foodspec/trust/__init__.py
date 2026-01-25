"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
Trust module: Uncertainty quantification, calibration, robustness checks.

Quantifying model uncertainty:
    from foodspec.trust import MondrianConformalClassifier, evaluate_abstention
    cp = MondrianConformalClassifier(model, target_coverage=0.9)
    cp.fit(X_train, y_train)
    cp.calibrate(X_cal, y_cal, bins=stage_bins)
    result = cp.predict_sets(X_test, bins=stage_bins_test)
    abstain = evaluate_abstention(resulting_proba, y_test, threshold=0.6)
    print(result.coverage, abstain.abstain_rate)
"""

from foodspec.trust.conformal import ConformalPredictionResult, MondrianConformalClassifier
from foodspec.trust.abstain import (
    AbstentionResult,
    evaluate_abstention,
    MaxProbAbstainer,
    ConformalSizeAbstainer,
    CombinedAbstainer,
)
from foodspec.trust.calibration import (
    expected_calibration_error,
    TemperatureScaler,
    IsotonicCalibrator,
    PlattCalibrator,
)
from foodspec.trust.evaluator import TrustEvaluator, TrustEvaluationResult
from foodspec.trust.reliability import (
    brier_score,
    compute_calibration_metrics,
    reliability_curve_data,
    top_class_confidence,
    CalibrationMetrics,
)
from foodspec.trust.coverage import (
    coverage_by_group,
    format_coverage_table,
    to_markdown,
    to_latex,
    check_coverage_guarantees,
    coverage_comparison,
    summarize_coverage,
)
from foodspec.trust.interpretability import (
    extract_linear_coefficients,
    top_k_features,
    coefficient_summary,
    to_markdown_coefficients,
    compare_coefficients,
)
from foodspec.trust.permutation import (
    permutation_importance,
    permutation_importance_with_names,
    top_k_important_features,
    compare_importances,
)
from foodspec.trust.marker_panel_link import link_marker_panel_explanations
from foodspec.trust.regulatory import (
    validate_reproducibility,
    integrity_checks,
    generate_trust_summary,
)

__all__ = [
    "MondrianConformalClassifier",
    "ConformalPredictionResult",
    "AbstentionResult",
    "evaluate_abstention",
    "MaxProbAbstainer",
    "ConformalSizeAbstainer",
    "CombinedAbstainer",
    "expected_calibration_error",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "TrustEvaluator",
    "TrustEvaluationResult",
    "brier_score",
    "compute_calibration_metrics",
    "reliability_curve_data",
    "top_class_confidence",
    "CalibrationMetrics",
    "coverage_by_group",
    "format_coverage_table",
    "to_markdown",
    "to_latex",
    "check_coverage_guarantees",
    "coverage_comparison",
    "summarize_coverage",
    "extract_linear_coefficients",
    "top_k_features",
    "coefficient_summary",
    "to_markdown_coefficients",
    "compare_coefficients",
    "permutation_importance",
    "permutation_importance_with_names",
    "top_k_important_features",
    "compare_importances",
    "link_marker_panel_explanations",
    "validate_reproducibility",
    "integrity_checks",
    "generate_trust_summary",
]
