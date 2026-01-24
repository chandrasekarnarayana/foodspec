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
    from foodspec.validation import train_test_split, cross_validate
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    scores = cross_validate(model, X, y, cv=5, seed=42)
"""

__all__ = []
