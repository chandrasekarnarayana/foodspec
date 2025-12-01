# Meat domain template

Questions this page answers
- How do I classify meat types/freshness with foodspec?
- How do the domain templates map to the core oil-auth workflow?
- What do CLI and Python runs look like?

## Use case and data
Assume a Raman/FTIR dataset of meat samples with label column `meat_type` (e.g., chicken, beef, pork) or freshness (`meat_label` such as fresh/spoiled). The meat template reuses the oil-auth pipeline (baseline, smoothing, normalization, cropping, peaks/ratios + classifier).

## Pipeline outline
1) CSV → HDF5 library (wide example):
```bash
foodspec csv-to-library data/meat.csv libraries/meat.h5 \
  --format wide \
  --wavenumber-column wavenumber \
  --label-column meat_type \
  --modality raman
```
2) Run classification via domain template (CLI):
```bash
foodspec domains \
  libraries/meat.h5 \
  --type meat \
  --label-column meat_type \
  --classifier-name rf \
  --output-dir runs/meat_demo
```
Outputs: CV metrics CSV, confusion_matrix.png, report.md, summary.json.

### Python example
```python
from foodspec.data import load_library
from foodspec.apps.meat import run_meat_authentication_workflow

fs = load_library("libraries/meat.h5")
res = run_meat_authentication_workflow(fs, label_column="meat_type", classifier_name="rf", cv_splits=5)
print(res.cv_metrics.head())
```

## Interpretation and reporting
- Look for balanced accuracy/macro F1 across meat classes; inspect confusion matrix for specific confusions (e.g., similar cuts).
- Report main: overall accuracy/macro F1 + confusion matrix; supplementary: per-class precision/recall, feature importances/ratios.
- Cross-validation reduces bias; mention folds/stratification.

See also
- [domains_overview.md](domains_overview.md)
- [oil_auth_tutorial.md](oil_auth_tutorial.md)
- [metrics_interpretation.md](metrics_interpretation.md)
- [methodsx_protocol.md](methodsx_protocol.md)
