# Reference Protocol — FoodSpec Standard Workflow

**Who:**
Food scientists, quality analysts, and researchers implementing FoodSpec in production or research settings.

**What:**
The canonical FoodSpec protocol: step-by-step workflow for data acquisition, preprocessing, model training, and validation of a food spectroscopy classification or regression task.

**When:**
Use this protocol as a template for any new FoodSpec analysis (oil authentication, adulterant detection, quality monitoring, etc.).

**When NOT:**
Do not use as a replacement for method-specific validation (ISO/regulatory) or when a domain-specific protocol exists (e.g., ISO/TS standards for oils).

**Key Assumptions:**
- Spectra are acquired on a calibrated Raman or FTIR instrument
- ≥30 samples per class; ≥3 replicates per sample
- Batch effects are documented and managed
- Reference or ground-truth labels are available for model training and validation

**What can go wrong:**
- Small training sets → overfitting and unreliable validation estimates
- Unmanaged batch effects → models that fail to generalize
- Data leakage (same sample in train/test) → inflated accuracy
- Preprocessing-dependent results → models that break if preprocessing changes

---

## Overview: The FoodSpec Standard Workflow

```
1. Study Design & Data Planning
   ↓
2. Sample Acquisition & Labeling
   ↓
3. Spectral Data Acquisition
   ↓
4. Data QC & Preprocessing
   ↓
5. Feature Extraction / Dimensionality Reduction
   ↓
6. Model Selection & Training (with Validation)
   ↓
7. Test Set Evaluation & Interpretation
   ↓
8. Deployment & Monitoring
```

---

## Step 1: Study Design & Data Planning

### Objectives & Hypotheses
Define a clear research question:
- **Classification:** "Can we distinguish authentic olive oils from counterfeit oils?"
- **Regression:** "Can we predict the oxidation level of oil samples?"

### Sample Size Calculation
Use power analysis (see [Study Design](../stats/study_design_and_data_requirements.md)):

```
Sample size per class (classification):
n ≥ max(
  1.96² × p(1-p) / (2 × α)²,     # Precision for proportion
  30                              # Minimum for ML
)

where p = expected effect proportion, α = acceptable error
```

**Rule of thumb:** ≥30 samples per class; ≥3 replicates per sample (total observations = n_classes × 30 × 3 = minimum).

### Batch & Confound Planning

- **Randomize batch order:** If analyzing samples across days/instruments, randomize assignment to batches
- **Include batch controls:** Same reference material scanned on every batch date
- **Document metadata:** Temperature, humidity, instrument settings, operator

### Definitions

Agree on:
- **Class definitions:** What makes an oil "authentic" vs. "adulterated"? (e.g., ≤2% adulterant = authentic)
- **Exclusion criteria:** Missing data, invalid spectra, contaminated samples
- **Replication:** What constitutes a "replicate"? (e.g., same sample, rescan on same day; or same vial, rescan on different day?)

---

## Step 2: Sample Acquisition & Labeling

### Sample Collection

1. **Source samples** from controlled (reference materials) and real (production, market) sources
2. **Create adulterant mixtures** if testing fraud detection (e.g., 1%, 2%, 5%, 10% adulterant)
3. **Store samples** under controlled conditions (cool, dark, sealed); document storage dates
4. **Record metadata:**
   - Sample ID, class/label, supplier, lot, storage conditions, acquisition date
   - For mixtures: composition and preparation method

### Ground-Truth Assignment

- Use orthogonal reference method (e.g., GC, HPLC, isotope ratio MS) OR expert consensus
- Record confidence in ground truth (e.g., "confirmed by GC" vs. "vendor claim")
- For novel adulterants: test with spiking/mixing experiments

---

## Step 3: Spectral Data Acquisition

### Instrument Setup

Choose one or both:
- **Raman:** Laser wavelength (532, 633, 785 nm), resolution, integration time
- **FTIR:** Resolution (4 cm⁻¹ standard), number of scans (32–64 recommended)

### Standard Operating Procedure (SOP)

```yaml
# Example FTIR SOP
Instrument:
  Type: FTIR (Perkin-Elmer/Bruker/etc.)
  Resolution: 4 cm⁻¹
  Wavenumber range: 400–4000 cm⁻¹
  Scans per spectrum: 32
  Background: Air, scanned every 10 samples

Sample Preparation:
  Amount: 1–2 µL (oils); 1–2 mg (solids)
  Substrate: ZnSe windows (oils) or KBr pellet (solids)
  Drying time: None (oils); 5–10 min (solids)

Data Collection:
  Temperature: 22 ± 2 °C
  Sample orientation: Consistent across replicates
  File format: .csv (wavenumber, absorbance) or instrument-native

QC:
  - Verify dark current (all zeros)
  - Verify background baseline (smooth, no spikes)
  - Check sample spectrum for saturation (no clipping)
```

### Replication Protocol

Acquire ≥3 replicates per sample:

| Replicate Level | Procedure | Use Case |
|-----------------|-----------|----------|
| **Technical** | Same vial, immediate rescans (3×) | Assess instrument noise |
| **Intra-day** | Same sample, rescans after re-mounting (3×) | Assess sample/mounting variability |
| **Inter-day** | Same sample, rescans on separate days (3×) | Assess temporal drift |
| **Total** | 9–27 spectra per sample | Recommended for new applications |

---

## Step 4: Data QC & Preprocessing

### Quality Checks

```
For each spectrum:
  ✓ No clipping (no intensities at detector max/min)
  ✓ SNR adequate (peak heights >> noise floor)
  ✓ Baseline reasonable (smooth, no extreme slopes)
  ✓ No cosmic rays or spikes (< 1 per 500 wavenumbers)
  ✓ Wavenumber range complete (no missing regions)

If failed:
  → Re-acquire or exclude from analysis
  → Document reason in metadata
```

### Preprocessing Pipeline

**Canonical order (apply in sequence):**

1. **Cosmic ray removal** (if Raman)
   - Automatic spike detection (e.g., `sklearn.preprocessing.SpectralCleaner`) or manual inspection
2. **Baseline correction**
   - Algorithm: Asymmetric Least Squares (ALS) or automatic baseline fitting
   - Rationale: Remove instrument offset and fluorescence
3. **Smoothing** (optional, if SNR low)
   - Savitzky–Golay filter (window=5–11, polynomial=2–3)
   - Target: Reduce noise without losing peak structure
4. **Normalization**
   - Standard: Min–max (0–1) or unit vector (L2)
   - Rationale: Make models scale-invariant
5. **Feature extraction** (optional, if using classical methods)
   - Peak heights, peak areas, peak ratios, or first/second derivatives
   - OR proceed to PCA/PLS without explicit feature engineering

**FoodSpec preprocessing config example:**

```python
preprocessing_config = {
    "baseline_correction": {
        "method": "als",
        "lambda": 100,
        "p": 0.01
    },
    "smoothing": {
        "method": "savgol",
        "window_length": 7,
        "polyorder": 2
    },
    "normalization": {
        "method": "unit_vector"
    },
    "feature_extraction": None  # Skip; use PLS on full spectrum
}
```

---

## Step 5: Feature Extraction / Dimensionality Reduction

### Options

| Method | Pros | Cons | When to use |
|--------|------|------|-------------|
| **PLS** | Supervised; fast; interpretable | Assumes linear relationships | Standard; most applications |
| **PCA** | Unsupervised; fast | No predictive power alone; linear | Exploratory; pre-screening |
| **Random Forest** | Non-linear; robust; no scaling needed | Black box; large feature space | Non-linear patterns; high-d |
| **Neural Network** | Non-linear; expressive | Requires more data; overfits easily | Large datasets (>500 samples); complex patterns |
| **SVM** | Non-linear (via kernel); data-efficient | Hyperparameter tuning required | Small-to-medium datasets with clear separation |

**Recommendation:** Start with PLS for interpretability. Use non-linear methods only if PLS insufficient and validation data adequate (n > 100).

---

## Step 6: Model Selection & Training (with Validation)

### Nested Cross-Validation

Use **nested CV** to avoid optimistic bias:

```
Outer loop (5-fold stratified CV):
  For each fold:
    Test set = 20% of data (held out)
    
    Inner loop (5-fold stratified CV on training set):
      Tune hyperparameters via grid search
      Select best hyperparameters
    
    Train final model on training set (best hyperparameters)
    Evaluate on test set
    
Record outer fold metrics (accuracy, AUC, RMSE, etc.)
Average across folds for unbiased estimate
```

### Hyperparameter Ranges

**PLS:**
- Components: 2–15

**Random Forest:**
- n_estimators: 50–500
- max_depth: 5–20
- min_samples_split: 2–10

**SVM:**
- C: 0.001–100 (log scale)
- kernel: 'rbf', 'poly', 'linear'
- gamma: 'scale', 'auto'

### Early Stopping Criteria

Stop tuning if:
- Validation metric plateaus (no improvement for 5 iterations)
- Computation time exceeds budget
- Overfitting detected (train metric >> validation metric)

---

## Step 7: Test Set Evaluation & Interpretation

### Reporting Metrics

**For classification:**

| Metric | Formula | When to use |
|--------|---------|------------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Balanced classes; easy interpretation |
| Precision | TP / (TP + FP) | Minimize false positives (e.g., false contamination alarms) |
| Recall | TP / (TP + FN) | Minimize false negatives (e.g., missed adulterants) |
| AUC-ROC | Area under ROC curve | Threshold-agnostic; compare models |
| F1-score | 2 × (Precision × Recall) / (Precision + Recall) | Balanced precision–recall tradeoff |

**For regression:**

| Metric | Formula |
|--------|---------|
| RMSE | sqrt(mean((y_true - y_pred)²)) |
| MAE | mean(\|y_true - y_pred\|) |
| R² | 1 - (SS_res / SS_tot) |

### Feature Importance

Report and interpret:

```
Method 1: PLS Loadings
  — Positive/negative loadings on principal components
  — Visualize as loading plots
  
Method 2: Permutation Importance
  — Shuffle each feature; measure drop in test metric
  — Identifies features that contribute to predictions
  
Method 3: SHAP Values
  — Model-agnostic feature attribution
  — Explains individual predictions
```

### Confidence Intervals & Error Bounds

Always report:

```
Point estimate ± 95% CI

Example:
  Accuracy: 94.2% (88.5%–97.1%)
  RMSE: 2.3 mg/kg ± 0.8
```

Compute CI via:
- **Bootstrap:** Resample test set with replacement; recompute metric; take 2.5th–97.5th percentile
- **Cross-validation:** Report range of fold metrics

---

## Step 8: Deployment & Monitoring

### Pre-Deployment Checklist

- [ ] Validation metrics acceptable (accuracy >85% OR domain-specific threshold)
- [ ] No signs of leakage (same sample in train/test)
- [ ] Batch effects managed (validation includes diverse batches)
- [ ] Feature importance reasonable (no single feature drives predictions)
- [ ] Error analysis complete (understand failure modes)
- [ ] Metadata documented (preprocessing params, training data, date)

### Deployment

1. **Retrain on full dataset** (if using nested CV, which holds out test set)
2. **Save model** with version number and training data hash
3. **Implement monitoring:**
   - Routine QC samples (reference materials) scanned with every batch
   - Model predictions tracked; alert if accuracy drops
   - Batch effect detection (e.g., SIMCA-class distance or Hotelling T²)

### Monitoring Metrics

```
For each new batch:
  
  1. QC spectrum predictions
     — Expected: Consistent predictions for known reference
     — Alert if: >2 SD deviation from expected
  
  2. Batch effect magnitude
     — Calculate: Mean distance of batch samples from training set
     — Alert if: Distance > 3 × training set SD
  
  3. Model age
     — Recommendation: Retrain every 6–12 months
     — Alert if: Data accumulates significantly
```

---

## When Results Cannot Be Trusted

🚨 **Critical red flags — stop and investigate:**

1. **Training metrics >> validation metrics** (train acc=99%, val acc=80%)
   - Likely cause: Overfitting; dataset too small; leakage
   - Action: Increase sample size; add regularization; check for leakage

2. **Perfect or near-perfect accuracy** (>98%) without domain explanation
   - Likely cause: Batch confounding; data leakage; artificial separation
   - Action: Examine confusion matrix; verify train/test independence; check feature importance

3. **Unstable CV folds** (fold 1: 95%, fold 2: 70%, fold 3: 88%)
   - Likely cause: Small test set per fold; outliers; imbalanced classes
   - Action: Increase sample size; use stratified CV; apply robust cross-validation

4. **Feature importance dominated by 1–2 features**
   - Likely cause: Confounding variable; instrument drift; batch effect
   - Action: Validate in independent experiment; include batch controls; investigate feature meaning

5. **Model fails on new batch/instrument**
   - Likely cause: Batch effects unmanaged during training; instrument shift
   - Action: Retrain with batch correction; use batch-aware CV; validate on diverse batches

---

## See Also

- [Study Design](../stats/study_design_and_data_requirements.md) — How to plan FoodSpec studies
- [Model Evaluation](../ml/model_evaluation_and_validation.md) — Validation metrics and interpretation
- [Workflows](../workflows/) — Domain-specific examples
- [Non-Goals and Limitations](../non_goals_and_limitations.md) — What FoodSpec cannot do

