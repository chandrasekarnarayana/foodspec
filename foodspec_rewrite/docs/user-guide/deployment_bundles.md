# Deployment Bundle Export and Import

The `foodspec.deploy` module provides utilities for creating and loading deployment bundles. A deployment bundle contains all necessary artifacts for model deployment, including the trained model, preprocessing pipeline, label encoder, and metadata.

## Bundle Contents

A deployment bundle includes:

- **Protocol snapshot**: Complete protocol configuration (JSON)
- **Preprocessing pipeline**: List of preprocessing steps with parameters (JSON)
- **Trained model**: Serialized scikit-learn model (joblib)
- **Label encoder**: Mapping from integer labels to class names (JSON)
- **Feature grid (x_grid)**: Wavenumber grid or feature names (NumPy array)
- **Metadata schema**: Expected metadata columns and types (JSON)
- **Package versions**: Versions of Python and key dependencies (JSON)
- **Bundle manifest**: Index of bundle contents (JSON)

## Saving a Bundle

Use `save_bundle()` to export a deployment bundle:

```python
from pathlib import Path
from foodspec.deploy import save_bundle
import numpy as np

protocol = {
    "version": "2.0.0",
    "data": {"modality": "raman"},
    "preprocessing": {"steps": [{"method": "snv"}]}
}

preprocess_pipeline = [{"method": "snv", "params": {}}]
label_encoder = {0: "authentic", 1: "adulterated"}
x_grid = np.linspace(400, 4000, 100)

bundle_path = save_bundle(
    run_dir=Path("./my_run"),
    bundle_dir=Path("./deployment_bundle"),
    protocol=protocol,
    preprocess_pipeline=preprocess_pipeline,
    model_path=Path("./my_run/model.joblib"),
    label_encoder=label_encoder,
    x_grid=x_grid,
    metadata_schema={"sample_id": "str", "temperature": "float"}
)
```

## Loading a Bundle

Use `load_bundle()` to import a deployment bundle:

```python
from foodspec.deploy import load_bundle

bundle = load_bundle(Path("./deployment_bundle"))

# Access components
protocol = bundle["protocol"]
model = bundle["model"]
label_encoder = bundle["label_encoder"]
x_grid = bundle["x_grid"]
preprocess_pipeline = bundle["preprocess_pipeline"]
metadata_schema = bundle["metadata_schema"]
versions = bundle["package_versions"]

# Make predictions
predictions = model.predict(X_test)
predicted_classes = [label_encoder[p] for p in predictions]
```

## Bundle Structure

A typical bundle directory contains:

```
deployment_bundle/
├── bundle_manifest.json       # Index of bundle contents
├── protocol.json               # Protocol configuration
├── preprocess_pipeline.json    # Preprocessing steps
├── model.joblib                # Trained model
├── label_encoder.json          # Class name mapping
├── x_grid.npy                  # Feature grid (NumPy array)
├── metadata_schema.json        # Metadata column types
└── package_versions.json       # Dependency versions
```

## Bundle Manifest

The `bundle_manifest.json` file tracks the bundle format version and contents:

```json
{
  "bundle_format_version": "1.0.0",
  "contents": {
    "protocol": "protocol.json",
    "preprocess_pipeline": "preprocess_pipeline.json",
    "model": "model.joblib",
    "label_encoder": "label_encoder.json",
    "x_grid": "x_grid.npy",
    "metadata_schema": "metadata_schema.json",
    "package_versions": "package_versions.json"
  }
}
```

## Example: Complete Workflow

```python
import tempfile
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from foodspec.deploy import save_bundle, load_bundle

# 1. Train model
X_train = np.random.randn(100, 50)
y_train = np.random.randint(0, 3, 100)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = Path("./model.joblib")
joblib.dump(model, model_path)

# 2. Create bundle
protocol = {"version": "2.0.0", "data": {"modality": "raman"}}
preprocess_pipeline = [
    {"method": "snv", "params": {}},
    {"method": "savgol", "params": {"window_length": 11, "polyorder": 2}}
]
label_encoder = {0: "class_a", 1: "class_b", 2: "class_c"}
x_grid = np.linspace(400, 4000, 50)

bundle_path = save_bundle(
    run_dir=Path("./run"),
    bundle_dir=Path("./bundle"),
    protocol=protocol,
    preprocess_pipeline=preprocess_pipeline,
    model_path=model_path,
    label_encoder=label_encoder,
    x_grid=x_grid
)

# 3. Load and use bundle
bundle = load_bundle(bundle_path)

X_test = np.random.randn(5, 50)
predictions = bundle["model"].predict(X_test)
predicted_classes = [bundle["label_encoder"][p] for p in predictions]

print(f"Predictions: {predictions}")
print(f"Classes: {predicted_classes}")
```

## API Reference

### `save_bundle()`

```python
save_bundle(
    run_dir: Path | str,
    bundle_dir: Path | str,
    protocol: Mapping[str, Any],
    preprocess_pipeline: Optional[List[Dict[str, Any]]] = None,
    model_path: Optional[Path | str] = None,
    label_encoder: Optional[Dict[int, str]] = None,
    x_grid: Optional[np.ndarray] = None,
    metadata_schema: Optional[Dict[str, str]] = None,
) -> Path
```

**Parameters:**

- `run_dir`: Source run directory (for consistency with artifact paths)
- `bundle_dir`: Target directory for the bundle
- `protocol`: Expanded protocol snapshot (from manifest or ProtocolV2)
- `preprocess_pipeline`: List of preprocessing step dictionaries
- `model_path`: Path to trained model file (will be copied into bundle)
- `label_encoder`: Mapping from integer labels to class names
- `x_grid`: Feature grid (wavenumbers or feature names)
- `metadata_schema`: Schema mapping canonical keys to column types

**Returns:**

- `bundle_path`: Path to the created bundle directory

### `load_bundle()`

```python
load_bundle(bundle_dir: Path | str) -> Dict[str, Any]
```

**Parameters:**

- `bundle_dir`: Path to the bundle directory

**Returns:**

- `bundle`: Dictionary with keys:
  - `protocol`: Protocol configuration
  - `preprocess_pipeline`: Preprocessing steps (or None)
  - `model`: Trained model (or None)
  - `label_encoder`: Class name mapping (or None)
  - `x_grid`: Feature grid (or None)
  - `metadata_schema`: Metadata schema (or None)
  - `package_versions`: Dependency versions
  - `manifest`: Bundle manifest

**Raises:**

- `FileNotFoundError`: If bundle manifest is missing
- `ValueError`: If bundle format is unsupported

## Version Compatibility

Bundle format version 1.0.0 is the initial release. Future versions will maintain backward compatibility or provide migration utilities.

The `package_versions.json` file records:

- Python version
- NumPy version
- pandas version
- SciPy version
- scikit-learn version
- Pydantic version

This enables version compatibility checks during deployment.

## Best Practices

1. **Always include preprocessing pipeline**: Ensures consistent preprocessing during inference
2. **Version bundles**: Use timestamps or semantic versioning in bundle directory names
3. **Test loaded bundles**: Verify predictions match expected behavior after loading
4. **Document custom components**: If using custom transformers, document reconstruction steps
5. **Check package versions**: Verify dependency compatibility in deployment environment

## Integration with FoodSpec Runs

After completing a FoodSpec run, create a bundle from the run artifacts:

```python
from pathlib import Path
from foodspec.core import load_manifest
from foodspec.deploy import save_bundle

# Load run manifest
run_dir = Path("./foodspec_runs/20250106_120000_oil_auth")
manifest = load_manifest(run_dir / "manifest.json")

# Create bundle
bundle_path = save_bundle(
    run_dir=run_dir,
    bundle_dir=Path("./bundles/oil_auth_v1"),
    protocol=manifest.expanded_protocol,
    model_path=run_dir / "artifacts" / "model.joblib",
    label_encoder={0: "authentic", 1: "adulterated"},
    x_grid=manifest.wavenumber_grid
)
```

## See Also

- [Model Training](../workflows/model_training.md)
- [Validation](../workflows/validation.md)
- [Artifact Registry](../reference/artifacts.md)
