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
    from foodspec.deploy import save_bundle, load_bundle, predict_from_bundle
    bundle_path = save_bundle(run_dir, bundle_dir, protocol=protocol, model_path=model_path)
    bundle = load_bundle(bundle_path)
    predictions = predict_from_bundle(bundle, input_csv="new_data.csv", output_dir="./results")
"""

from .bundle import load_bundle, save_bundle
from .predict import predict_from_bundle, predict_from_bundle_path

__all__ = ["save_bundle", "load_bundle", "predict_from_bundle", "predict_from_bundle_path"]


__all__ = []
