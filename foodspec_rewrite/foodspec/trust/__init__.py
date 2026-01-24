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
    from foodspec.trust import confidence_interval, calibration_curve
    ci = confidence_interval(model, X_test, y_test, seed=42)
    cal_prob, true_prob = calibration_curve(model, X_test, y_test)
"""

__all__ = []
