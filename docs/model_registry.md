# Model Registry

The model registry utilities let you persist trained pipelines along with metadata for reproducibility and deployment.

## Programmatic usage

```python
from foodspec.model_registry import save_model, load_model

# Save
save_model(
    model=pipeline,
    path="models/oil_rf_v1",
    name="oil_auth_pipeline",
    version="1.0.0",
foodspec_version="0.2.0",
    extra={"label_column": "oil_type"},
)

# Load
model, metadata = load_model("models/oil_rf_v1")
print(metadata)
```

## Using the CLI

Save a model while running a workflow and inspect it later:

```bash
foodspec oil-auth ./out/preprocessed.h5 --label-column oil_type --output-report ./out/report.html --save-model ./models/oil_rf_v1
foodspec model-info ./models/oil_rf_v1
```

The base path (`./models/oil_rf_v1`) is used for both `.joblib` and `.json` artifacts. The `model-info` command prints the stored metadata (name, version, foodspec version, timestamp, and any extra fields).
