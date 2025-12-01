# Oil authentication workflow

Questions this page answers
- Can we classify edible oils from Raman/FTIR spectra?
- How do I run the workflow via CSV → library → oil-auth?
- How do I interpret PCA, confusion matrix, and feature importance?
- How should I report the results?

## Scientific question
Determine oil type and detect adulteration using vibrational spectra.

## End-to-end example
### CLI path
1) Convert CSV to library (wide format example):
```bash
foodspec csv-to-library data/oils.csv libraries/oils.h5 \
  --format wide \
  --wavenumber-column wavenumber \
  --label-column oil_type \
  --modality raman
```
2) Run oil authentication (CLI):
```bash
foodspec oil-auth libraries/oils.h5 \
  --label-column oil_type \
  --classifier-name rf \
  --cv-splits 5 \
  --output-dir runs/oils_demo
```
Outputs: metrics.json/CSV, confusion_matrix.png, feature importances (if available), report.md.

### Python variant
```python
from foodspec.data import load_library
from foodspec.apps.oils import run_oil_authentication_workflow

fs = load_library("libraries/oils.h5")
result = run_oil_authentication_workflow(fs, label_column="oil_type", classifier_name="rf", cv_splits=5)
print(result.cv_metrics.head())
```

### Inspecting PCA and confusion matrix
- PCA: visualize clustering of classes; look for separation by oil type.
- Confusion matrix: check misclassifications; identify similar oils being confused.
- Feature importance: peak/ratio contributions; focus on chemically meaningful bands (e.g., 1655/1742).

## Interpretation
- Accuracy/macro F1 reflect overall performance; use stratified CV (default).  
- Good accuracy is dataset-dependent; for clean lab spectra expect higher scores; for challenging/adulterated sets, report limitations.
- Cross-validation reduces optimistic bias and shows variance across folds.

## Reporting
- Main text: overall accuracy/macro F1, confusion matrix figure, brief preprocessing/model description.  
- Supplementary: per-class precision/recall/F1, feature importances/ratios, spectra examples, run metadata/configs.
- For MethodsX/FAIR: include preprocessing steps (baseline, smoothing, normalization, crop), classifier choice, CV design, and dataset provenance.

## Optional: comparing ratios between oil types with statistical tests
You can test whether a specific band ratio differs across oil types (useful for interpretation/papers).
```python
import pandas as pd
from scipy.stats import kruskal
from foodspec.features.ratios import compute_ratios

# Assume peak heights already extracted to df_peaks; here we build a ratio
ratio_def = {"ratio_1655_1745": ("peak_1655.0_height", "peak_1742.0_height")}
df_ratios = compute_ratios(df_peaks, ratio_def)
df_ratios["oil_type"] = fs.metadata["oil_type"].values

groups = [g["ratio_1655_1745"].values for _, g in df_ratios.groupby("oil_type")]
stat, p = kruskal(*groups)
print(f"Kruskal–Wallis H={stat:.3f}, p={p:.3g}")
```
Interpretation: a small p-value suggests at least one oil type has a different ratio distribution. Not required for classification, but useful for scientific interpretation and MethodsX-style reporting.

See also
- [csv_to_library.md](csv_to_library.md)
- [metrics_interpretation.md](metrics_interpretation.md)
- [keyword_index.md](keyword_index.md)
- [ftir_raman_preprocessing.md](ftir_raman_preprocessing.md)
