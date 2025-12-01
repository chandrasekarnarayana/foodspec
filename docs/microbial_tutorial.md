# Microbial domain template

Questions this page answers
- How do I classify microbial spectra with foodspec?
- How does the microbial template map to the core workflows?
- What do CLI and Python runs look like?

## Use case and data
Assume spectral data with label column `species` or `strain` (or contamination `status`: positive/negative). The microbial template reuses the oil-auth pipeline (baseline, smoothing, normalization, cropping, peaks/ratios + classifier).

## Pipeline outline
1) CSV → HDF5 library:
```bash
foodspec csv-to-library data/microbial.csv libraries/microbial.h5 \
  --format wide \
  --wavenumber-column wavenumber \
  --label-column species \
  --modality raman
```
2) Run classification via domain template (CLI):
```bash
foodspec domains \
  libraries/microbial.h5 \
  --type microbial \
  --label-column species \
  --classifier-name svm_rbf \
  --output-dir runs/microbial_demo
```
Outputs: CV metrics CSV, confusion_matrix.png, report.md, summary.json.

### Python example
```python
from foodspec.data import load_library
from foodspec.apps.microbial import run_microbial_detection_workflow

fs = load_library("libraries/microbial.h5")
res = run_microbial_detection_workflow(fs, label_column="species", classifier_name="svm_rbf", cv_splits=5)
print(res.cv_metrics.head())
```

## Considerations
- Class imbalance is common; rely on macro/weighted F1 and per-class metrics.
- Cross-validation and, ideally, external validation (independent isolates) are important for robustness.

## Reporting
- Main: overall accuracy/macro F1, confusion matrix; note handling of imbalance.  
- Supplementary: per-class precision/recall/F1, spectra examples, feature importances/ratios, run metadata/configs.

See also
- [domains_overview.md](domains_overview.md)
- [metrics_interpretation.md](metrics_interpretation.md)
- [oil_auth_tutorial.md](oil_auth_tutorial.md)
- [methodsx_protocol.md](methodsx_protocol.md)
