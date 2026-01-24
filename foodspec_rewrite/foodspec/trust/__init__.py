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
from foodspec.trust.abstain import AbstentionResult, evaluate_abstention

__all__ = [
    "MondrianConformalClassifier",
    "ConformalPredictionResult",
    "AbstentionResult",
    "evaluate_abstention",
]
