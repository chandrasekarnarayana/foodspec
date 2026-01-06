# Model registry

Questions this page answers
- How do I save and load trained foodspec pipelines?
- What metadata is stored?
- How do I apply a saved model in a new session?

## What metadata is stored?
- `name`, `version`, `foodspec_version`
- Timestamp
- Model artifact (`.joblib`) + metadata JSON
- `extra`: label/model configuration, any custom fields (e.g., classifier name, label column)

## End-to-end example (Python)
Train (e.g., oil-auth), save, then load and apply to new data.
```python
from foodspec import load_library
from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.model_registry import save_model, load_model

# Train on a library
fs_train = load_library("libraries/oils_train.h5")
res = run_oil_authentication_workflow(fs_train, label_column="oil_type", classifier_name="rf", cv_splits=5)

# Save pipeline
save_model(
    model=res.pipeline,
    path="models/oil_rf_v1",
    name="oil_rf",
    version="0.2.0",
    foodspec_version="0.2.0",
    extra={"label_column": "oil_type", "classifier_name": "rf"},
)

# Load and apply to new samples
model_loaded, meta = load_model("models/oil_rf_v1")
fs_new = load_library("libraries/oils_new.h5")
preds = model_loaded.predict(fs_new.x)
print(meta, preds[:5])
```

## CLI flow
- Save during workflow:
```bash
foodspec oil-auth libraries/oils_train.h5 \
  --label-column oil_type \
  --classifier-name rf \
  --save-model models/oil_rf_v1 \
  --output-dir runs/oils_rf
```
- Inspect metadata:
```bash
foodspec model-info models/oil_rf_v1
```
To apply the saved model, load it in Python (as above) and predict on a new HDF5 library.

## Best practices
- Version models (e.g., v1, v2) and record foodspec_version.
- Store training data summary (labels, date, preprocessing choices) in `extra`.
- Keep artifacts (`.joblib` + `.json`) under version control or a model registry; retain for QA/regulatory traceability.

See also
- `cli.md`
- `metrics/metrics_and_evaluation/`

- `oil_auth_tutorial.md`
