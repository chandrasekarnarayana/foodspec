"""Package for machine learning utilities and fusion strategies."""

from foodspec.ml.calibration import (
    CalibrationDiagnostics,
    calibration_slope_intercept,
    compute_calibration_diagnostics,
    recalibrate_classifier,
)
from foodspec.ml.fusion import (
    DecisionFusionResult,
    LateFusionResult,
    decision_fusion_vote,
    decision_fusion_weighted,
    late_fusion_concat,
)
from foodspec.ml.lifecycle import (
    ModelAgingScore,
    ModelLifecycleTracker,
    ModelState,
    PerformanceSnapshot,
    SunsetRule,
)

__all__ = [
    # Fusion
    "DecisionFusionResult",
    "LateFusionResult",
    "decision_fusion_vote",
    "decision_fusion_weighted",
    "late_fusion_concat",
    # Calibration
    "CalibrationDiagnostics",
    "calibration_slope_intercept",
    "compute_calibration_diagnostics",
    "recalibrate_classifier",
    # Lifecycle
    "ModelAgingScore",
    "ModelLifecycleTracker",
    "ModelState",
    "PerformanceSnapshot",
    "SunsetRule",
]

__all__ = [
    "LateFusionResult",
    "DecisionFusionResult",
    "late_fusion_concat",
    "decision_fusion_vote",
    "decision_fusion_weighted",
]
