# Workflow: Oil Authentication

> If you are new to designing spectral workflows, see [Designing & reporting workflows](workflow_design_and_reporting.md).
> For model choices and evaluation guidance, see [ML & DL models](../ml/models_and_best_practices.md) and [Metrics & evaluation](../metrics/metrics_and_evaluation.md).

Oil authentication addresses “What oil is this?” and “Is it adulterated?” using Raman/FTIR spectra. This workflow provides a complete, reproducible recipe from raw spectra to publication-ready metrics and plots.

Relevant visual aids: spectrum overlays, PCA scores/loadings, confusion matrix, boxplots/violin plots of key ratios. See [Plots guidance](workflow_design_and_reporting.md#plots-visualizations) for expectations.

```mermaid
flowchart LR
  A[Raw spectra (HDF5/CSV)] --> B[Preprocessing (baseline, Savitzky–Golay, norm, crop)]
  B --> C[Feature extraction (peaks + ratios)]
  C --> D[Model training (e.g., RF/SVM/PLS-DA)]
  D --> E[Evaluation (CV metrics, confusion matrix, PCA)]
  E --> F[Reports (plots, metrics.json, report.md)]
```

## 1. Problem and dataset
- **Why labs care:** Adulteration (cheap oils in EVOO), mislabeling, batch verification.
- **Inputs:** Spectral library (HDF5) with columns: `oil_type` (label), optional `batch`, `instrument`. Wavenumber axis in ascending cm⁻¹ (Raman/FTIR fingerprint 600–1800 cm⁻¹, CH stretch 2800–3100 cm⁻¹).
- **Typical size:** Tens to hundreds of spectra per class for robust models; synthetic examples work for testing.

## 2. Pipeline (default)
- **Preprocessing:** ALS baseline → Savitzky–Golay smoothing → L2 normalization → crop to 600–1800 cm⁻¹.
- **Features:** Expected peaks (≈1655 C=C, 1742 C=O, 1450 CH2 bend); ratios (1655/1742, 1450/1655).
- **Models:** Random Forest (robust default), or SVM/PLS-DA for linear boundaries.
- **Validation:** Stratified k-fold CV (default 5 folds); metrics: accuracy, macro F1; confusion matrix.

## 3. Python example
```python
from foodspec.data.loader import load_example_oils
from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.chemometrics.pca import run_pca
from foodspec.viz.pca import plot_pca_scores

fs = load_example_oils()
result = run_oil_authentication_quickstart(fs, label_column="oil_type")

# Metrics
print(result.cv_metrics)

# Confusion matrix
fig_cm = plot_confusion_matrix(result.confusion_matrix, result.class_labels)
fig_cm.savefig("oil_confusion.png", dpi=150)

# PCA visualization on preprocessed spectra
pca, res = run_pca(fs.x, n_components=2)
fig_scores = plot_pca_scores(res.scores, labels=fs.metadata["oil_type"])
fig_scores.savefig("oil_pca.png", dpi=150)
```

### Optional deep-learning baseline
```python
# pip install foodspec[deep]
from foodspec.chemometrics.deep import Conv1DSpectrumClassifier
from foodspec.metrics import compute_classification_metrics

model = Conv1DSpectrumClassifier(n_filters=8, n_epochs=15, batch_size=16, random_state=0)
model.fit(fs.x, fs.metadata["oil_type"])
dl_pred = model.predict(fs.x)
dl_metrics = compute_classification_metrics(fs.metadata["oil_type"], dl_pred)
print("DL accuracy:", dl_metrics["accuracy"])
```
Use only when you have sufficient samples per class; always compare against classical baselines and inspect F1_macro/confusion matrices.

## 4. CLI example (with config)
Create `examples/configs/oil_auth_quickstart.yml`:
```yaml
input_hdf5: libraries/oils.h5
label_column: oil_type
classifier_name: rf
cv_splits: 5
```
Run:
```bash
foodspec oil-auth --config examples/configs/oil_auth_quickstart.yml --output-dir runs/oil_auth_demo
```
Outputs: `metrics.json`, `confusion_matrix.png`, `report.md` in a timestamped folder.

## 5. Interpretation (MethodsX tone)
- Report overall accuracy and macro F1; include confusion matrix with class labels.
- Mention preprocessing steps (baseline, smoothing, normalization, crop) and feature choices (peak/ratio definitions).
- Highlight chemically meaningful loadings/feature importances (e.g., unsaturation bands).
- Main text: summary metrics + confusion matrix figure. Supplement: per-class precision/recall, spectra examples, configs.

## Summary
- Baseline + smoothing + normalization + crop → peak/ratio features → RF/SVM/PLS-DA → CV metrics and confusion matrix.
- Use stratified CV; report macro metrics; tie discriminative bands back to chemistry.

## Statistical analysis
- **Why:** Beyond classification metrics, test whether key ratios differ across oil types to support interpretation.
- **Example (ANOVA + Tukey):**
```python
import pandas as pd
from foodspec.stats import run_anova, run_tukey_hsd
from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.data.loader import load_example_oils
import pandas as pd

fs = load_example_oils()
res = run_oil_authentication_quickstart(fs, label_column="oil_type")

# Extract ratio features from the fitted pipeline
preproc = res.pipeline.named_steps["preprocess"]
features = res.pipeline.named_steps["features"]
feat_array = features.transform(preproc.transform(fs.x))
cols = features.named_steps["to_array"].columns_
ratio_series = pd.Series(feat_array[:, 0], index=fs.metadata.index, name=cols[0])

anova_res = run_anova(ratio_series, fs.metadata["oil_type"])
print(anova_res.summary)
try:
    tukey = run_tukey_hsd(ratio_series, fs.metadata["oil_type"])
    print(tukey.head())
except ImportError:
    pass
```
- **Interpretation:** ANOVA p-value < 0.05 suggests at least one oil type differs in the ratio; Tukey identifies which pairs. Report effect size where possible.
See theory: [Hypothesis testing](../stats/hypothesis_testing_in_food_spectroscopy.md), [ANOVA](../stats/anova_and_manova.md).

## Further reading
- [Baseline correction](../preprocessing/baseline_correction.md)
- [Feature extraction](../preprocessing/feature_extraction.md)
- [Classification & regression](../ml/classification_regression.md)
- [Model evaluation](../ml/model_evaluation_and_validation.md)
