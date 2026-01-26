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

from foodspec.trust.conformal import ConformalPredictionResult, MondrianConformalClassifier, ConformalPredictor
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
from foodspec.trust.metrics import (
    compute_calibration_metrics as compute_trust_calibration_metrics,
    negative_log_likelihood,
    risk_coverage_curve,
    bootstrap_ci,
    bootstrap_coverage_efficiency,
)
from foodspec.trust.abstention import (
    apply_abstention_rules,
    filter_importance_by_acceptance,
)
from foodspec.trust.readiness import RegulatoryReadiness, compute_readiness_score
from foodspec.trust.schema import (
    CalibrationArtifact,
    ConformalArtifact,
    AbstentionArtifact,
    ReadinessArtifact,
)
try:
    from foodspec.trust.evaluator import TrustEvaluator, TrustEvaluationResult
except ImportError:
    # Evaluator requires foodspec.core.artifacts which may not be available
    TrustEvaluator = None
    TrustEvaluationResult = None
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
from foodspec.trust.dataset_cards import DatasetCard, write_dataset_card
from foodspec.trust.model_cards import ModelCard, write_model_card

__all__ = [
    "MondrianConformalClassifier",
    "ConformalPredictionResult",
    "ConformalPredictor",
    "AbstentionResult",
    "evaluate_abstention",
    "MaxProbAbstainer",
    "ConformalSizeAbstainer",
    "CombinedAbstainer",
    "expected_calibration_error",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "compute_trust_calibration_metrics",
    "negative_log_likelihood",
    "risk_coverage_curve",
    "bootstrap_ci",
    "bootstrap_coverage_efficiency",
    "apply_abstention_rules",
    "filter_importance_by_acceptance",
    "RegulatoryReadiness",
    "compute_readiness_score",
    "CalibrationArtifact",
    "ConformalArtifact",
    "AbstentionArtifact",
    "ReadinessArtifact",
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
    "ModelCard",
    "DatasetCard",
    "write_model_card",
    "write_dataset_card",
]
