# Artifact Versioning & Compatibility

## Overview
FoodSpec `.foodspec` artifacts embed the version of the package that created them. Load-time compatibility checks prevent using an artifact with an incompatible FoodSpec version.

## Version Metadata
Each artifact stores:
- **`foodspec_version`**: the FoodSpec version (e.g., `0.2.1`) that created the artifact
- **`artifact_schema_version`**: the schema version of the artifact structure (e.g., `1.0`)

## Compatibility Rules

### Major Version Mismatch
If the artifact's major version differs from your FoodSpec major version (e.g., artifact built with `0.2.1`, loading with `1.0.0`), `load_artifact()` will raise `ValueError`. This prevents silent model mismatches.

**Solution:** Re-export the model with the current FoodSpec version.

### Minor/Patch Differences
- Patch-level differences (e.g., `0.2.1` → `0.2.3`) are allowed.
- Minor-level increases (e.g., `0.2.1` → `0.3.0`) are allowed (forward-compatible).

## Override Compatibility Checks
For emergency/legacy scenarios, bypass version guards:

```python
from foodspec import load_artifact

# Allow loading incompatible artifact (NOT recommended for production)
predictor = load_artifact("old_model.foodspec", allow_incompatible=True)
```

## Best Practices

1. **Version Control:** Store the FoodSpec version alongside your model.
   ```bash
   foodspec --version
   ```

2. **Automate Re-Export:** When upgrading FoodSpec major versions, re-train and export models.
   ```python
   from foodspec import FoodSpec, save_artifact
   fs = FoodSpec(...)
   save_artifact(fs.bundle, "model_v1_0_0.foodspec")
   ```

3. **Docker/Environment:** Pin FoodSpec version in `requirements.txt` or `environment.yml` for reproducibility.
   ```yaml
   # environment.yml
   dependencies:
     - python=3.11
     - pip
     - pip:
       - foodspec==0.2.1
   ```

## See Also
- [Artifact Save/Load](artifact_export.md)
- [Model Registry](model_registry.md)
- [Reproducibility](../repro/run_record.md)
