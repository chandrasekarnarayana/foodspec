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
Models module: ML model wrappers for sklearn, XGBoost, Keras.

Training and predicting with models:
    from foodspec.models import RandomForestModel
    model = RandomForestModel(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

__all__ = []
