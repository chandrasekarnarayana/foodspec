# Phase 1 API Reference

Complete API documentation for FoodSpec Phase 1 unified entry point and core objects.

## Table of Contents

1. [FoodSpec (Entry Point)](#foodspec-entry-point)
2. [Spectrum (Core Data Model)](#spectrum-core-data-model)
3. [RunRecord (Provenance)](#runrecord-provenance)
4. [OutputBundle (Artifact Management)](#outputbundle-artifact-management)

---

## FoodSpec (Entry Point)

**Import**: `from foodspec import FoodSpec`

Unified entry point for spectroscopy workflows. Accepts multiple input formats and provides chainable API for QC → preprocessing → feature extraction → training → export.

### Constructor

```python
FoodSpec(
    source: str | Path | np.ndarray | pd.DataFrame | FoodSpectrumSet,
    wavenumbers: np.ndarray | None = None,
    metadata: pd.DataFrame | None = None,
    modality: str = "raman",
    kind: str | None = None,
    output_dir: str | Path | None = None,
)
```

**Parameters**:
- `source`: Data source. Auto-detects format:
  - `str/Path`: CSV file or folder of spectra
  - `np.ndarray`: Shape (n_samples, n_features), will wrap in FoodSpectrumSet
  - `pd.DataFrame`: Wide format (rows=samples, cols=wavenumbers)
  - `FoodSpectrumSet`: Direct use
- `wavenumbers`: Optional x-axis values. Auto-loaded from CSV if not provided.
- `metadata`: Optional metadata DataFrame. Auto-loaded from CSV if not provided.
- `modality`: Spectroscopy type ("raman", "ftir", "nir")
- `kind`: Domain label ("oils", "dairy", "heating", etc.)
- `output_dir`: Directory for export outputs (default: current directory)

**Returns**: FoodSpec instance with internal `data` (FoodSpectrumSet) and `bundle` (OutputBundle)

**Example**:
```python
# From CSV
fs = FoodSpec("oils.csv")

# From numpy array
fs = FoodSpec(
    np.random.randn(50, 500),
    wavenumbers=np.linspace(500, 2000, 500),
    metadata=pd.DataFrame({"label": ["A", "B"] * 25})
)

# From DataFrame
fs = FoodSpec(df_spectra, wavenumbers=wn)

# From FoodSpectrumSet
fs = FoodSpec(existing_spectrum_set)
```

### Chainable Methods

#### `.qc(method="isolation_forest", threshold=0.5, **kwargs) -> FoodSpec`

Quality control: detect and optionally remove outlier spectra.

**Parameters**:
- `method`: Outlier detection method
  - `"isolation_forest"`: Sklearn IsolationForest (default)
  - `"zscore"`: Z-score based detection
- `threshold`: Contamination fraction (0-1) or z-score cutoff
- `**kwargs`: Additional arguments for the detection method

**Returns**: self (for chaining)

**Side Effects**:
- Removes outliers from `self.data` if any detected
- Records step in `self.bundle.run_record`
- Adds "outlier_scores" to diagnostics
- Adds "outliers_detected" metric

**Example**:
```python
fs.qc(method="isolation_forest", threshold=0.1)  # Remove bottom 10% as outliers
```

#### `.preprocess(preset="standard", **kwargs) -> FoodSpec`

Apply preprocessing pipeline (Phase 1: step logging only, full integration in Phase 2).

**Parameters**:
- `preset`: Preset name (Phase 1 options):
  - `"quick"`: Fast preprocessing
  - `"standard"`: Balanced preprocessing (default)
  - `"publication"`: Rigorous preprocessing
- `**kwargs`: Override preset parameters

**Returns**: self (for chaining)

**Side Effects**:
- Records step in `self.bundle.run_record`
- Tracks preprocessing in `self._steps_applied`

**Example**:
```python
fs.preprocess("standard")
fs.preprocess("publication", smooth_window=7)
```

#### `.features(preset="standard", **kwargs) -> FoodSpec`

Extract features: peaks, ratios, statistical features, etc. (Phase 1: step logging only, full integration in Phase 4).

**Parameters**:
- `preset`: Preset name
- `**kwargs`: Additional parameters

**Returns**: self (for chaining)

**Side Effects**:
- Records step in `self.bundle.run_record`
- Tracks feature extraction in `self._steps_applied`

**Example**:
```python
fs.features("oil_auth")
fs.features("standard", n_peaks=10)
```

#### `.train(algorithm="rf", label_column=None, cv_folds=5, **kwargs) -> FoodSpec`

Train a machine learning model (Phase 1: step logging only, full integration in Phase 2).

**Parameters**:
- `algorithm`: Algorithm name
  - `"rf"`: Random Forest (default)
  - `"lr"`: Logistic Regression
  - `"svm"`: Support Vector Machine
  - `"pls_da"`: PLS-DA
- `label_column`: Metadata column with labels
- `cv_folds`: Number of cross-validation folds (default: 5)
- `**kwargs`: Additional parameters for the algorithm

**Returns**: self (for chaining)

**Side Effects**:
- Records step in `self.bundle.run_record`
- Tracks training in `self._steps_applied`
- Stores model in `self.bundle.artifacts`

**Example**:
```python
fs.train("rf", label_column="oil_type", cv_folds=10)
fs.train("svm", label_column="class", C=0.1)
```

#### `.export(output_dir=None, formats=["json", "csv"]) -> Path`

Export all outputs (metrics, diagnostics, artifacts, provenance) to disk.

**Parameters**:
- `output_dir`: Export directory (uses `self.output_dir` if None)
- `formats`: List of export formats
  - `"json"`: JSON format (default)
  - `"csv"`: CSV format (for DataFrames)
  - `"png"`: PNG format (for matplotlib figures)
  - `"pdf"`: PDF format (for matplotlib figures)
  - `"joblib"`: Joblib format (for models)
  - `"pickle"`: Pickle format (for objects)

**Returns**: Path to output directory

**Output Structure**:
```
output_dir/
  metrics/
    metrics.json              # All metrics as JSON
    <metric_name>.csv         # DataFrame metrics as CSV
  diagnostics/
    <diagnostic_name>.json    # Dict diagnostics
    <diagnostic_name>.csv     # DataFrame diagnostics
    <diagnostic_name>.png     # Plot diagnostics
  artifacts/
    <artifact_name>.joblib    # Model artifacts
    <artifact_name>.pickle    # Object artifacts
  provenance.json             # RunRecord serialized
```

**Example**:
```python
out_dir = fs.export()                                    # Uses default output_dir
out_dir = fs.export("./results/")                       # Explicit directory
out_dir = fs.export("./results/", formats=["json", "csv", "png"])
```

### Other Methods

#### `.summary() -> str`

Generate human-readable workflow summary.

**Returns**: String describing dataset, steps applied, and output statistics

**Example**:
```python
print(fs.summary())
# Output:
# FoodSpec Workflow Summary
# ==================================================
# Dataset: raman, n=50, n_features=500
# Steps applied: qc, preprocess(standard), train(rf)
# 
# OutputBundle(run_id=foodspec_20240101T120)
#   Metrics: 5 items
#   Diagnostics: 3 items
#   Artifacts: 2 items
```

### Properties

#### `.data: FoodSpectrumSet`
The underlying spectroscopy dataset.

#### `.bundle: OutputBundle`
Container for all outputs (metrics, diagnostics, artifacts, provenance).

#### `.config: dict`
Configuration parameters for reproducibility.

---

## Spectrum (Core Data Model)

**Import**: `from foodspec import Spectrum`

Represents a single spectroscopic measurement with validated metadata.

### Constructor

```python
Spectrum(
    x: np.ndarray,
    y: np.ndarray,
    kind: str = "raman",
    x_unit: str = "cm-1",
    metadata: dict | None = None,
)
```

**Parameters**:
- `x`: Spectral axis (wavenumbers or wavelengths), shape (n_features,)
- `y`: Intensity values, shape (n_features,)
- `kind`: Spectroscopy type ("raman", "ftir", "nir")
- `x_unit`: Unit of x-axis ("cm-1", "nm", "um", "1/cm")
- `metadata`: Arbitrary metadata dict (sample_id, batch, etc.)

**Example**:
```python
spec = Spectrum(
    x=np.linspace(500, 2000, 500),
    y=np.random.randn(500),
    kind="raman",
    metadata={"sample_id": "oil_001", "batch": 1}
)
```

### Methods

#### `.normalize(method="vector") -> Spectrum`

Normalize intensity values.

**Parameters**:
- `method`: Normalization method
  - `"vector"`: Vector norm (L2 norm)
  - `"max"`: Max absolute value
  - `"area"`: Area under curve

**Returns**: New Spectrum with normalized y values

**Example**:
```python
spec_norm = spec.normalize("vector")
spec_max = spec.normalize("max")
```

#### `.crop_wavenumber(x_min, x_max) -> Spectrum`

Extract specified wavenumber range.

**Parameters**:
- `x_min`: Minimum wavenumber (inclusive)
- `x_max`: Maximum wavenumber (inclusive)

**Returns**: New Spectrum with cropped x and y values

**Example**:
```python
spec_cropped = spec.crop_wavenumber(600, 1600)
```

#### `.copy() -> Spectrum`

Create a deep copy.

**Returns**: New Spectrum instance with copied arrays

**Example**:
```python
spec2 = spec.copy()
```

### Properties

#### `.config_hash: str`
SHA256 hash of metadata for reproducibility. Two Spectrum objects with identical metadata will have the same config_hash.

#### `.n_points: int`
Number of spectral points (length of x and y arrays).

---

## RunRecord (Provenance)

**Import**: `from foodspec import RunRecord`

Immutable record of workflow execution with complete provenance tracking.

### Constructor

```python
RunRecord(
    workflow_name: str,
    config: dict,
    dataset_hash: str,
    user: str | None = None,
    notes: str | None = None,
)
```

**Parameters**:
- `workflow_name`: Name of workflow (e.g., "oil_authentication")
- `config`: Configuration parameters as dict
- `dataset_hash`: SHA256 hash of input data
- `user`: Optional user identifier
- `notes`: Optional workflow notes

**Example**:
```python
record = RunRecord(
    workflow_name="oil_authentication",
    config={"algorithm": "rf", "n_estimators": 100},
    dataset_hash="8b5f18a1",
    user="analyst@example.com",
    notes="Validation run 2024-01-01"
)
```

### Methods

#### `.add_step(name, step_hash, error=None, metadata=None) -> None`

Record a workflow step with reproducible hash.

**Parameters**:
- `name`: Step name (e.g., "preprocessing", "training")
- `step_hash`: SHA256 hash of step inputs/outputs
- `error`: Optional error message if step failed
- `metadata`: Optional metadata dict for the step

**Example**:
```python
record.add_step("preprocessing", "abc123def456", metadata={"baseline": "als"})
record.add_step("training", "xyz789uvw012", metadata={"cv_folds": 5})
```

#### `.to_json(path: str | Path) -> None`

Serialize to JSON file.

**Parameters**:
- `path`: Output file path

**Example**:
```python
record.to_json("run_20240101_120000.json")
```

#### `.from_json(path: str | Path) -> RunRecord` (classmethod)

Deserialize from JSON file.

**Parameters**:
- `path`: Input JSON file path

**Returns**: RunRecord instance

**Example**:
```python
loaded_record = RunRecord.from_json("run_20240101_120000.json")
```

### Properties

#### `.config_hash: str`
SHA256 hash of config parameters (first 8 characters).

#### `.dataset_hash: str`
SHA256 hash of input data array.

#### `.combined_hash: str`
SHA256 hash of config + dataset + all steps (first 8 characters). Uniquely identifies the workflow execution.

#### `.run_id: str`
Unique run identifier: `{workflow_name}_{timestamp}`

#### `.environment: dict`
System environment info:
- `python_version`: Python version string
- `platform`: OS platform
- `hostname`: Machine hostname
- `packages`: Dict of package versions (numpy, pandas, scikit-learn, etc.)

#### `.step_records: list`
List of executed steps. Each step is a dict with:
- `name`: Step name
- `hash`: Step hash
- `timestamp`: Execution timestamp
- `error`: Error message (if failed)
- `metadata`: Step-specific metadata

---

## OutputBundle (Artifact Management)

**Import**: `from foodspec import OutputBundle`

Container for all workflow outputs: metrics, diagnostics, artifacts, and provenance.

### Constructor

```python
OutputBundle(
    run_record: RunRecord,
    output_dir: str | Path | None = None,
)
```

**Parameters**:
- `run_record`: RunRecord instance for provenance
- `output_dir`: Optional default output directory

**Example**:
```python
record = RunRecord("oil_auth", {}, "hash123")
bundle = OutputBundle(run_record=record, output_dir="./results/")
```

### Methods

#### `.add_metrics(name: str, value: Any) -> None`

Add a metric (scalar or DataFrame).

**Parameters**:
- `name`: Metric name
- `value`: Scalar, array, DataFrame, or other serializable object

**Example**:
```python
bundle.add_metrics("accuracy", 0.95)
bundle.add_metrics("cv_scores", pd.DataFrame({...}))
bundle.add_metrics("confusion_matrix", np.array([[...], [...]]))
```

#### `.add_diagnostic(name: str, value: Any) -> None`

Add a diagnostic output (plot, array, DataFrame, dict).

**Parameters**:
- `name`: Diagnostic name
- `value`: Any serializable object (matplotlib figure, array, DataFrame, dict)

**Example**:
```python
bundle.add_diagnostic("roc_curve", roc_auc_df)
bundle.add_diagnostic("feature_importance", feature_importance_array)
bundle.add_diagnostic("confusion_matrix", confusion_matrix_fig)
```

#### `.add_artifact(name: str, value: Any) -> None`

Add an artifact (model, preprocessor, etc.).

**Parameters**:
- `name`: Artifact name
- `value`: Any serializable object (sklearn model, preprocessor, custom object)

**Example**:
```python
bundle.add_artifact("model", trained_rf_model)
bundle.add_artifact("scaler", preprocessing_scaler)
```

#### `.export(output_dir=None, formats=["json", "csv"]) -> Path`

Export all outputs to disk in multiple formats.

**Parameters**:
- `output_dir`: Export directory (uses `self.output_dir` if None)
- `formats`: List of export formats to use
  - `"json"`: JSON (default)
  - `"csv"`: CSV (for DataFrames)
  - `"png"`: PNG (for matplotlib figures)
  - `"pdf"`: PDF (for matplotlib figures)
  - `"joblib"`: Joblib (for sklearn models)
  - `"pickle"`: Pickle (for general objects)

**Returns**: Path to output directory

**Output Structure**:
```
output_dir/
  metrics/
    metrics.json              # All metrics
    <metric_name>.csv         # Each DataFrame metric
  diagnostics/
    <diagnostic_name>.json    # Dict diagnostics
    <diagnostic_name>.csv     # DataFrame diagnostics
    <diagnostic_name>.png     # Figure diagnostics
  artifacts/
    <artifact_name>.joblib    # sklearn models
    <artifact_name>.pickle    # Other artifacts
  provenance.json             # RunRecord
```

**Example**:
```python
out_dir = bundle.export("./results/")
out_dir = bundle.export("./results/", formats=["json", "csv", "png", "joblib"])
```

#### `.summary() -> str`

Generate summary of outputs.

**Returns**: Human-readable summary

**Example**:
```python
print(bundle.summary())
# OutputBundle Summary
# ==================================================
# Metrics: 5 items
#   - accuracy: 0.95
#   - f1: 0.93
#   - cv_scores: (5, 2) DataFrame
# 
# Diagnostics: 2 items
#   - confusion_matrix: (3, 3) array
#   - roc_curve: Figure
# 
# Artifacts: 1 item
#   - model: RandomForestClassifier
# 
# Provenance: foodspec_20240101T120
```

### Properties

#### `.metrics: dict`
All added metrics as dictionary.

#### `.diagnostics: dict`
All added diagnostics as dictionary.

#### `.artifacts: dict`
All added artifacts as dictionary.

#### `.run_record: RunRecord`
Provenance record for the workflow.

---

## Complete Workflow Example

```python
from foodspec import FoodSpec
import numpy as np
import pandas as pd

# 1. Create synthetic data
x = np.random.randn(50, 500)
wn = np.linspace(500, 2000, 500)
metadata = pd.DataFrame({
    "sample_id": [f"s{i:03d}" for i in range(50)],
    "oil_type": ["olive", "sunflower", "canola"] * 16 + ["olive"],
})

# 2. Initialize FoodSpec
fs = FoodSpec(
    x,
    wavenumbers=wn,
    metadata=metadata,
    modality="raman",
    kind="oils",
)

# 3. Execute chainable workflow
fs.qc()                                    # Quality control
  .preprocess("standard")                  # Preprocessing
  .features("oil_auth")                    # Feature extraction
  .train("rf", label_column="oil_type")   # Model training
  .export("./results/")                   # Export outputs

# 4. Access results
print(fs.bundle.metrics["accuracy"])       # Get accuracy metric
print(fs.summary())                        # Print workflow summary

# 5. Provenance
print(fs.bundle.run_record.combined_hash)  # Full workflow hash
for step in fs.bundle.run_record.step_records:
    print(f"{step['name']}: {step['hash']}")
```

---

## Type Hints

All classes are fully type-hinted for IDE autocompletion and static type checking:

```python
from foodspec import FoodSpec, Spectrum, RunRecord, OutputBundle
from pathlib import Path
import numpy as np
import pandas as pd

# Full type safety
fs: FoodSpec = FoodSpec("data.csv")
spec: Spectrum = Spectrum(
    x=np.array([1, 2, 3]),
    y=np.array([1, 2, 3])
)
record: RunRecord = fs.bundle.run_record
bundle: OutputBundle = fs.bundle
metrics: dict = bundle.metrics
```

---

## Error Handling

All classes include validation and clear error messages:

```python
# Shape validation in Spectrum
try:
    spec = Spectrum(x=np.array([1, 2]), y=np.array([1, 2, 3]))
    # ValueError: x and y must have same length
except ValueError as e:
    print(f"Error: {e}")

# Kind validation
try:
    spec = Spectrum(x=np.array([1, 2]), y=np.array([1, 2]), kind="xray")
    # ValueError: kind must be one of: raman, ftir, nir
except ValueError as e:
    print(f"Error: {e}")
```

---

## Performance Notes

- **Spectrum.normalize()**: O(n) where n = number of points
- **Spectrum.crop()**: O(n log n) for binary search + slicing
- **RunRecord.add_step()**: O(1) list append
- **OutputBundle.export()**: O(k) where k = number of outputs
  - JSON serialization: ~1ms per 10KB
  - CSV export: ~5ms per 10000 rows
  - Pickle: ~2ms per MB

For large datasets (>10GB), consider:
- Using HDF5 storage (Phase 2)
- Batch processing (split data into chunks)
- Lazy loading (FoodSpectrumSet supports iterator mode)

---

## Compatibility

- Python: 3.8+
- numpy: 1.19+
- pandas: 1.1+
- scikit-learn: 0.23+ (for IsolationForest in FoodSpec.qc())
- matplotlib: 3.0+ (for figure export)

---

## See Also

- [PHASE1_COMPLETION.md](PHASE1_COMPLETION.md) - Full implementation summary
- [PHASE1_IMPLEMENTATION_SUMMARY.md](PHASE1_IMPLEMENTATION_SUMMARY.md) - Architecture details
- [examples/phase1_quickstart.py](examples/phase1_quickstart.py) - Live example
- [tests/test_phase1_core.py](tests/test_phase1_core.py) - Test cases
