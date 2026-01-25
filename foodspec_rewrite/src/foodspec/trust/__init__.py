"""
Trust & Uncertainty Quantification subsystem.

Provides:
- Conformal prediction with Mondrian conditioning
- Probability calibration (temperature scaling, isotonic regression)
- Principled abstention and selective classification
- Group-aware coverage analysis
- High-level evaluator integration
"""

from foodspec.trust.conformal import (
    MondrianConformalClassifier,
    ConformalPredictionResult,
)
from foodspec.trust.calibration import (
    expected_calibration_error,
    TemperatureScaler,
    IsotonicCalibrator,
)
from foodspec.trust.abstain import (
    evaluate_abstention,
    AbstentionResult,
)
from foodspec.trust.evaluator import (
    TrustEvaluator,
    TrustEvaluationResult,
)

__all__ = [
    "MondrianConformalClassifier",
    "ConformalPredictionResult",
    "expected_calibration_error",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "evaluate_abstention",
    "AbstentionResult",
    "TrustEvaluator",
    "TrustEvaluationResult",
]
