# Data governance imports
from foodspec.qc.dataset_qc import check_class_balance, diagnose_imbalance

# Drift detection for production monitoring
from foodspec.qc.drift import (
    DriftReport as ProductionDriftReport,
)
from foodspec.qc.drift import (
    detect_drift as detect_production_drift,
)
from foodspec.qc.drift import (
    detect_feature_drift,
    kl_divergence,
    population_stability_index,
    should_recalibrate,
)
from foodspec.qc.engine import (
    DriftResult,
    HealthResult,
    OutlierResult,
    QCReport,
    compute_health_scores,
    detect_drift,
    detect_outliers,
    generate_qc_report,
)
from foodspec.qc.leakage import detect_batch_label_correlation, detect_replicate_leakage
from foodspec.qc.novelty import novelty_score_single, novelty_scores
from foodspec.qc.prediction_qc import PredictionQCResult, evaluate_prediction_qc
from foodspec.qc.readiness import compute_readiness_score
from foodspec.qc.replicates import assess_variability_sources, compute_replicate_consistency

__all__ = [
    # Existing QC engine
    "QCReport",
    "DriftResult",
    "HealthResult",
    "OutlierResult",
    "compute_health_scores",
    "detect_drift",
    "detect_outliers",
    "generate_qc_report",
    # Data governance
    "check_class_balance",
    "diagnose_imbalance",
    "compute_replicate_consistency",
    "assess_variability_sources",
    "detect_batch_label_correlation",
    "detect_replicate_leakage",
    "compute_readiness_score",
    "novelty_scores",
    "novelty_score_single",
    # Production drift monitoring
    "ProductionDriftReport",
    "detect_production_drift",
    "detect_feature_drift",
    "kl_divergence",
    "population_stability_index",
    "should_recalibrate",
    # Prediction QC
    "PredictionQCResult",
    "evaluate_prediction_qc",
]
