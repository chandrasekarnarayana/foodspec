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
Deploy module: Model serving, batch prediction, API deployment.

Deploying and serving models:
    from foodspec.deploy import PredictionServer, batch_predict
    server = PredictionServer(model_path="./model.pkl")
    predictions = batch_predict(model, new_data, batch_size=32)
"""

__all__ = []
