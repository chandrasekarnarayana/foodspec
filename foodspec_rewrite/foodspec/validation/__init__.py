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
Validation module: Cross-validation, train/test splits, evaluation metrics.

Validating model performance:
    from foodspec.validation import LeaveOneGroupOutSplitter, StratifiedKFoldOrGroupKFold
    splitter = StratifiedKFoldOrGroupKFold(n_splits=5, seed=42)
    for train_idx, test_idx in splitter.split(X, y, groups):
        ...
"""
from foodspec.validation.splits import (
    LeaveOneGroupOutSplitter,
    StratifiedKFoldOrGroupKFold,
)
from foodspec.validation.metrics import (
    accuracy,
    macro_f1,
    precision_macro,
    recall_macro,
    auroc_macro,
    expected_calibration_error,
    compute_classification_metrics,
    ece,  # Legacy API
    brier,  # Additional calibration metric
)
from foodspec.validation.evaluation import (
    EvaluationRunner,
    EvaluationResult,
    evaluate_model_cv,
    evaluate_model_nested_cv,
)
from foodspec.validation.statistics import (
    bootstrap_ci,
    anova_on_metric,
    manova_placeholder,
)

__all__ = [
    "LeaveOneGroupOutSplitter",
    "StratifiedKFoldOrGroupKFold",
    "accuracy",
    "macro_f1",
    "precision_macro",
    "recall_macro",
    "auroc_macro",
    "expected_calibration_error",
    "compute_classification_metrics",
    "ece",
    "brier",
    "EvaluationRunner",
    "EvaluationResult",
    "bootstrap_ci",
    "anova_on_metric",
    "manova_placeholder",
    "evaluate_model_cv",
    "evaluate_model_nested_cv",
]
