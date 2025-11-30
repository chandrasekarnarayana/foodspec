# QC & Novelty Detection

Quality control often requires screening spectra for outliers or suspect batches.

## Workflow

```python
import pandas as pd
from foodspec.apps.qc import train_qc_model, apply_qc_model
from foodspec.data.loader import load_example_oils

spectra = load_example_oils()
# Define authentic-only mask from metadata if available
train_mask = spectra.metadata["oil_type"] == "olive"  # example filter

model = train_qc_model(spectra, train_mask=train_mask, model_type="oneclass_svm")

qc_result = apply_qc_model(spectra, model=model)
print(qc_result.labels_pred.value_counts())
print(qc_result.threshold)
```

Interpretation:
- `labels_pred`: "authentic" vs "suspect" decisions per sample.
- `scores`: anomaly/decision scores; higher means more normal if `higher_score_is_more_normal=True`.
- `threshold`: decision threshold; override to adjust strictness.
