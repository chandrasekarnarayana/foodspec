# Inference-Only Prediction from Bundles

The `foodspec.deploy.predict` module provides utilities for making predictions on new data using deployment bundles. This enables inference-only workflows without requiring the full FoodSpec training pipeline.

## Overview

The prediction workflow:

1. **Load a deployment bundle** (created with `save_bundle()`)
2. **Prepare new data** in CSV format with spectral measurements
3. **Make predictions** using `predict_from_bundle()`
4. **Retrieve results** from output CSV files (predictions and probabilities)

## Quick Start

```python
from pathlib import Path
from foodspec.deploy import load_bundle, predict_from_bundle

# Load bundle
bundle = load_bundle(Path("./deployment_bundle"))

# Make predictions
predictions_df = predict_from_bundle(
    bundle=bundle,
    input_csv="new_samples.csv",
    output_dir="./predictions"
)

# View results
print(predictions_df)
```

## Input Data Format

Input CSV files must contain three columns:

- **`sample_id`**: Unique identifier for each sample
- **`wavenumber`**: Spectral feature position (wavenumber, wavelength, etc.)
- **`intensity`**: Spectral intensity value

Example CSV structure:

```csv
sample_id,wavenumber,intensity
SAMPLE_001,400.0,0.523
SAMPLE_001,401.5,0.531
SAMPLE_001,403.0,0.542
...
SAMPLE_002,400.0,0.612
SAMPLE_002,401.5,0.619
...
```

Each sample should have multiple rows (one per wavenumber) representing its complete spectrum.

## Custom Column Names

If your CSV uses different column names, specify them explicitly:

```python
predictions_df = predict_from_bundle(
    bundle=bundle,
    input_csv="data.csv",
    output_dir="./output",
    sample_id_col="id",
    wavenumber_col="freq",
    intensity_col="value"
)
```

## Output Files

The prediction function generates two CSV files:

### 1. `predictions.csv`

Contains the predicted class and label for each sample:

```csv
sample_id,predicted_class,predicted_label
SAMPLE_001,0,authentic
SAMPLE_002,1,adulterated
SAMPLE_003,0,authentic
```

- **`sample_id`**: Sample identifier
- **`predicted_class`**: Integer class label (0, 1, 2, ...)
- **`predicted_label`**: Human-readable class name (from label encoder)

### 2. `probabilities.csv`

Contains the probability distribution across all classes:

```csv
sample_id,prob_authentic,prob_adulterated_oil_a,prob_adulterated_oil_b
SAMPLE_001,0.987,0.010,0.003
SAMPLE_002,0.123,0.654,0.223
SAMPLE_003,0.891,0.089,0.020
```

- **`sample_id`**: Sample identifier
- **`prob_<class_name>`**: Probability for each class (sums to 1.0 per sample)

Set `save_probabilities=False` to skip generating this file.

## Preprocessing Pipeline

If the bundle contains a preprocessing pipeline, it will be automatically applied to input spectra before prediction:

```python
# Bundle with preprocessing
bundle_path = save_bundle(
    ...
    preprocess_pipeline=[
        {"method": "snv", "params": {}},
        {"method": "savgol", "params": {"window_length": 11, "polyorder": 2}},
    ]
)

# Predictions automatically apply SNV + Savitzky-Golay
predictions_df = predict_from_bundle(bundle, input_csv, output_dir)
```

Supported preprocessing methods:
- `snv`: Standard Normal Variate
- `savgol` / `savitzky_golay`: Savitzky-Golay smoothing
- `derivative`: Derivative transformation
- `vector_normalize`: Vector normalization
- `msc`: Multiplicative Scatter Correction
- `emsc`: Extended Multiplicative Scatter Correction
- `asls`: Asymmetric Least Squares baseline correction
- `despike` / `hampel`: Hampel despiking filter

## Complete Example

```python
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from foodspec.deploy import save_bundle, load_bundle, predict_from_bundle

# 1. Create and save model bundle
X_train = np.random.randn(100, 50)
y_train = np.random.randint(0, 3, 100)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

model_path = Path("model.joblib")
joblib.dump(model, model_path)

bundle_dir = Path("./bundle")
save_bundle(
    run_dir=Path("./run"),
    bundle_dir=bundle_dir,
    protocol={"version": "2.0.0"},
    preprocess_pipeline=[{"method": "snv", "params": {}}],
    model_path=model_path,
    label_encoder={0: "authentic", 1: "adulterated_a", 2: "adulterated_b"},
    x_grid=np.linspace(400, 4000, 50)
)

# 2. Create test data CSV
test_data = []
for sample_idx in range(5):
    for feature_idx in range(50):
        test_data.append({
            "sample_id": f"TEST_{sample_idx}",
            "wavenumber": 400 + feature_idx * 72,
            "intensity": np.random.randn()
        })
pd.DataFrame(test_data).to_csv("test_data.csv", index=False)

# 3. Load bundle and predict
bundle = load_bundle(bundle_dir)
predictions_df = predict_from_bundle(
    bundle=bundle,
    input_csv="test_data.csv",
    output_dir="./predictions"
)

# 4. View results
print(predictions_df)

# Load probabilities
prob_df = pd.read_csv("./predictions/probabilities.csv")
print(prob_df)
```

## Convenience Function

Use `predict_from_bundle_path()` to load the bundle and predict in one call:

```python
from foodspec.deploy import predict_from_bundle_path

predictions_df = predict_from_bundle_path(
    bundle_dir="./bundle",
    input_csv="test_data.csv",
    output_dir="./predictions"
)
```

This is equivalent to calling `load_bundle()` followed by `predict_from_bundle()`.

## API Reference

### `predict_from_bundle()`

```python
predict_from_bundle(
    bundle: Dict[str, Any],
    input_csv: Path | str,
    output_dir: Path | str,
    wavenumber_col: str = "wavenumber",
    intensity_col: str = "intensity",
    sample_id_col: str = "sample_id",
    save_probabilities: bool = True,
) -> pd.DataFrame
```

**Parameters:**

- `bundle`: Loaded bundle from `load_bundle()`
- `input_csv`: Path to input CSV file with spectra
- `output_dir`: Directory for output CSV files
- `wavenumber_col`: Column name for wavenumber values
- `intensity_col`: Column name for intensity values
- `sample_id_col`: Column name for sample identifiers
- `save_probabilities`: Whether to save probability matrix to CSV

**Returns:**

- `predictions_df`: DataFrame with `sample_id`, `predicted_class`, and `predicted_label`

**Raises:**

- `ValueError`: If bundle does not contain trained model or input CSV is missing required columns
- `FileNotFoundError`: If input CSV file does not exist

### `predict_from_bundle_path()`

```python
predict_from_bundle_path(
    bundle_dir: Path | str,
    input_csv: Path | str,
    output_dir: Path | str,
    wavenumber_col: str = "wavenumber",
    intensity_col: str = "intensity",
    sample_id_col: str = "sample_id",
    save_probabilities: bool = True,
) -> pd.DataFrame
```

Convenience function that loads the bundle and makes predictions in one call. Parameters and returns are the same as `predict_from_bundle()`, except `bundle_dir` replaces `bundle`.

## Error Handling

The prediction functions provide actionable error messages:

```python
# Missing model in bundle
>>> predictions_df = predict_from_bundle(empty_bundle, "data.csv", "./out")
ValueError: Bundle does not contain a trained model

# Missing input file
>>> predictions_df = predict_from_bundle(bundle, "missing.csv", "./out")
FileNotFoundError: Input CSV not found: missing.csv

# Missing required columns
>>> predictions_df = predict_from_bundle(bundle, "bad_data.csv", "./out")
ValueError: Input CSV missing required columns: ['intensity']

# Unknown preprocessing method
>>> # Bundle with invalid preprocessing step
ValueError: Unknown preprocessing method 'invalid_method'. Available methods: ...
```

## Best Practices

1. **Validate input data**: Ensure CSV has all required columns and proper format
2. **Check bundle completeness**: Verify bundle contains model before deploying
3. **Monitor predictions**: Inspect probabilities.csv for low-confidence predictions
4. **Handle preprocessing errors**: Catch exceptions when applying preprocessing pipeline
5. **Version bundles**: Track which bundle version generated predictions
6. **Log predictions**: Record prediction timestamps and bundle versions for audit trail

## Integration with Production Systems

### Batch Processing

```python
import glob
from pathlib import Path

bundle = load_bundle("./production_bundle")

for input_file in glob.glob("data_to_process/*.csv"):
    output_dir = Path("results") / Path(input_file).stem
    predict_from_bundle(bundle, input_file, output_dir)
    print(f"Processed {input_file} -> {output_dir}")
```

### API Endpoint

```python
from fastapi import FastAPI, UploadFile
from foodspec.deploy import load_bundle, predict_from_bundle

app = FastAPI()
bundle = load_bundle("./model_bundle")

@app.post("/predict")
async def predict(file: UploadFile):
    # Save uploaded CSV
    input_path = f"/tmp/{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # Make predictions
    predictions_df = predict_from_bundle(
        bundle, input_path, "/tmp/predictions", save_probabilities=False
    )
    
    # Return predictions as JSON
    return predictions_df.to_dict(orient="records")
```

## See Also

- [Deployment Bundles](deployment_bundles.md) - Creating and loading bundles
- [Model Training](../workflows/model_training.md) - Training models
- [Preprocessing](../reference/preprocessing.md) - Preprocessing methods
