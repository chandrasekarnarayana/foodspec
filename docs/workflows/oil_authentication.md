# Workflow: Oil Authentication

> If you are new to designing spectral workflows, see [Designing & reporting workflows](workflow_design_and_reporting.md).
> For model choices and evaluation guidance, see [ML & DL models](../ml/models_and_best_practices.md) and [Metrics & evaluation](../metrics/metrics_and_evaluation.md).

Oil authentication addresses “What oil is this?” and “Is it adulterated?” using Raman/FTIR spectra. This workflow provides a complete, reproducible recipe from raw spectra to report-ready metrics and plots.

Relevant visual aids: spectrum overlays, PCA scores/loadings, confusion matrix, boxplots/violin plots of key ratios. See [Plots guidance](workflow_design_and_reporting.md#plots-visualizations) for expectations. The tutorial content is merged here; see also the legacy tutorial file for reference.

```mermaid
flowchart LR
  subgraph Data
    A[Raw/vendor files]
    B[read_spectra -> FoodSpectrumSet]
  end
  subgraph Preprocess & Features
    C[Baseline + SavGol + norm + crop]
    D[Peaks + ratios]
  end
  subgraph Model
    E[RF / SVM / PLS-DA]
  end
  subgraph Evaluate
    F[CV metrics + confusion + PCA]
    G[Stats on ratios (ANOVA/G-H)]
  end
  subgraph Report
    H[plots + metrics.json + report.md]
  end
  A --> B --> C --> D --> E --> F --> H
  D --> G --> H
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

## 5. Interpretation
- Report overall accuracy and macro F1; include confusion matrix with class labels.
- Mention preprocessing steps (baseline, smoothing, normalization, crop) and feature choices (peak/ratio definitions).
- Highlight chemically meaningful loadings/feature importances (e.g., unsaturation bands).
- Main text: summary metrics + confusion matrix figure. Supplement: per-class precision/recall, spectra examples, configs.

### Qualitative & quantitative interpretation
- **Qualitative:** PCA scores and ratio boxplots show class structure; confusion matrix reveals which oils are confused. RF importances/PLS loadings (see interpretability figures) highlight bands driving separation—link back to unsaturation/carbonyl bands.
- **Quantitative:** Report macro F1/balanced accuracy; silhouette on PCA scores; ANOVA/Tukey/Games–Howell on key ratios (link to [ANOVA/MANOVA](../stats/anova_and_manova.md)); effect sizes when applicable.
- **Reviewer phrasing:** “PCA shows moderate separation of oil classes (silhouette ≈ …); the RF classifier reached macro F1 = …; ratios at 1655/1742 cm⁻¹ differed across oils (ANOVA p < …).”

### Peak & ratio summary tables
- Generate mean ± std of key peak positions/intensities and ratios by oil_type for supplementary tables.
- Example: use `compute_peak_stats` and `compute_ratio_table` on extracted features; report which bands/ratios differ most across oils (with p-values/effect sizes).
- Reviewer phrasing: “Table 1 summarizes unsaturation/carbonyl ratios by oil type (mean ± SD); Games–Howell indicates oil A > oil B (p_adj < …).”
- Visuals to pair: RF feature importances / PLS loadings (assets `rf_feature_importance.png`, `pls_loadings.png`) to link discriminative bands to chemistry.

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
# Robust post-hoc if variances/group sizes differ
gh = games_howell(ratio_series, fs.metadata["oil_type"])
print(gh.head())
```
- **Interpretation:** ANOVA p-value < 0.05 suggests at least one oil type differs in the ratio; Tukey or Games–Howell identifies which pairs. Report effect size where possible.
See theory: [Hypothesis testing](../stats/hypothesis_testing_in_food_spectroscopy.md), [ANOVA](../stats/anova_and_manova.md).

### Ratio plots (recommended)
- Use `plot_ratio_by_group` for key ratios (e.g., 1655/1742) across oil types; separated medians/IQRs imply differences—support with ANOVA/Games–Howell and effect sizes.
- Ratio–ratio scatter (e.g., 1655/1742 vs 3010/2850) highlights compositional regimes; pair with silhouette/ANOVA on each ratio.
- Summary tables (peak/ratio mean ± SD by oil_type) can accompany plots in supplementary material.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for oil authentication workflow:**

1. **Model trained and tested on oils from same source/batch (e.g., all "olive" from single producer/harvest)**
   - Intra-source variability unknown; model may learn producer-specific patterns, not species
   - Different olive cultivar or origin will fail
   - **Fix:** Include multiple sources per oil type; validate across different cultivars/origins

2. **No adulterant testing (model validated only on pure oils, not blends or refined oils)**
   - Pure-oil classification doesn't confirm ability to detect adulteration
   - Refined oils may cluster closer to pure oils than expected
   - **Fix:** Include known adulterants (refined oils, blends) in test set; test detection rates at 1%, 5%, 10% adulteration levels

3. **Ratios or features cherry-picked post-hoc to separate oils**
   - Data-dependent feature selection inflates reproducibility claims
   - Different dataset may reveal different separating features
   - **Fix:** Use univariate feature selection a priori; or use model-based importance from cross-validation

4. **Authentication model based on single spectral region (only CH stretches, ignore C=O region)**
   - Narrow spectral window may miss adulterants affecting other regions
   - Real adulterants exploit regions unchecked
   - **Fix:** Use full spectral range; test sensitivity to adulterants in different regions

5. **Cross-contamination during sample preparation (using same pipette for different oils)**
   - Cross-contamination creates false similarity between oils
   - Baseline or preprocessing steps may not remove contamination
   - **Fix:** Use separate equipment per sample; measure blanks between samples; document sample handling

6. **Confusing near-infrared (NIR) with Raman/FTIR without method validation**
   - Different spectroscopic methods give different spectral signatures
   - Transferring models between methods requires retraining
   - **Fix:** Validate method-specific models; don't mix spectra from different instruments/wavelengths without harmonization

7. **Model accuracy high (>95%) but specificity/sensitivity per oil type varies wildly**
   - Macro-accuracy can mask severe class-specific failures
   - Confusion matrix and per-class metrics reveal true performance
   - **Fix:** Report per-class precision/recall; show confusion matrix; investigate misclassified oils

8. **No temporal validation (model trained on 2024 oils, deployed on 2023 samples without revalidation)**
   - Aging, storage, or oxidation changes oil spectra over time
   - Model trained on recent oils may fail on archived samples
   - **Fix:** Test on samples from different harvest years; monitor model performance over time; retrain periodically

## Further reading
- [Baseline correction](../preprocessing/baseline_correction.md)
- [Feature extraction](../preprocessing/feature_extraction.md)
- [Classification & regression](../ml/classification_regression.md)
- [Model evaluation](../ml/model_evaluation_and_validation.md)
