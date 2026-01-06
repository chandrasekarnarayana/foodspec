# FoodSpec Flagship Examples Teaching Plan

**Purpose:** Identify 5 canonical teaching examples that represent core FoodSpec capabilities.  
**Target Audience:** Learners progressing from beginner to intermediate proficiency.  
**Approach:** Each example combines script + figures + datasets to teach a distinct capability.

---

## 1. Oil Authentication (Classification)

**Teaching Goal:** Demonstrate supervised classification workflow with peak-based features and cross-validation metrics.

### Source Script
- **File:** `examples/oil_authentication_quickstart.py`
- **Length:** ~35 lines
- **Capabilities Demonstrated:**
  - Loading example datasets (`load_example_oils()`)
  - Running oil authentication with quickstart API
  - Plotting confusion matrix (classification performance)
  - PCA visualization for exploratory analysis

### Existing Figures
- `docs/assets/figures/oil_confusion.png` (0 bytes, needs regeneration)
- `docs/assets/figures/oil_discriminative.png` (0 bytes, needs regeneration)
- `docs/assets/figures/oil_minimal_panel.png` (0 bytes, needs regeneration)
- `docs/assets/confusion_matrix_example.png` (34 KB, generic example)
- `docs/assets/pca_scores.png` (94 KB, generic PCA visualization)

### Dataset
- **Primary:** `examples/data/oil_synthetic.csv`
  - Structure: 8 samples (VO, PO, OO, CO × heating stages 0–1)
  - Columns: `oil_type`, `matrix`, `heating_stage`, `batch`, wavenumber columns (1000–1030)
  - Format: Wide format (1 row per sample)
- **Fallback:** Built-in `load_example_oils()` function

### Teaching Narrative
```
Step 1: Load example oils (3 types: VO, PO, OO, CO)
Step 2: Preprocess (baseline correction, normalization, peak extraction)
Step 3: Train classifier with cross-validation
Step 4: Evaluate with confusion matrix
Step 5: Visualize separability with PCA scores
Step 6: Export metrics summary (precision, recall, F1)
```

### Enhancement Needed
1. Regenerate `oil_confusion.png`, `oil_discriminative.png`, `oil_minimal_panel.png` from script output
2. Update script docstring with expected metrics outputs
3. Add inline comments explaining peak thresholds

---

## 2. Heating Stability / Quality Monitoring (Trend Analysis)

**Teaching Goal:** Demonstrate time-series analysis, peak ratio tracking, and trend modeling for quality assessment.

### Source Script
- **File:** `examples/heating_quality_quickstart.py`
- **Length:** ~35 lines
- **Capabilities Demonstrated:**
  - Creating/loading time-series spectroscopy data (`synthetic_heating_dataset()`)
  - Extracting key ratios via `run_heating_quality_workflow()`
  - Fitting trend models (degradation curves)
  - Plotting ratio vs. time with trend overlay

### Existing Figures
- `docs/assets/workflows/heating_quality_monitoring/heating_ratio_vs_time.png` (43 KB, **exists**)
- `docs/assets/figures/heating_trend.png` (0 bytes, needs regeneration)
- `docs/assets/figures/oil_stability.png` (0 bytes, needs regeneration)

### Dataset
- **Primary:** `examples/data/oil_synthetic.csv` with time-series interpretation
  - Column: `heating_stage` (0, 1, ..., N as time points)
  - Synthetic Raman with known degradation pattern
- **Built-in:** `synthetic_heating_dataset()` function

### Teaching Narrative
```
Step 1: Load heating dataset (Raman spectra at multiple heating times)
Step 2: Extract key ratio (e.g., I_1742 / I_2720) for each time point
Step 3: Fit trend model (linear, exponential, polynomial)
Step 4: Plot ratio curve with confidence bands
Step 5: Assess stability (slope, residual, R²)
Step 6: Report critical degradation thresholds
```

### Enhancement Needed
1. Ensure `heating_ratio_vs_time.png` is properly generated and documented
2. Add plot annotation (e.g., degradation rate equation, threshold line)
3. Include metadata: "Raman heating-time dataset, 4 oils × 2 time points"

---

## 3. Mixture Analysis (Quantification via NNLS)

**Teaching Goal:** Demonstrate unmixing/quantification workflow using non-negative least squares on synthetic mixtures.

### Source Script
- **File:** `examples/mixture_analysis_quickstart.py`
- **Length:** ~50 lines
- **Capabilities Demonstrated:**
  - Creating synthetic pure-component spectra
  - Generating mixture spectra with known fractions
  - Running NNLS unmixing (`run_mixture_analysis_workflow()`)
  - Comparing predicted vs. true coefficients
  - Assessing residual errors

### Existing Figures
- No dedicated figures currently; would benefit from:
  - Pure spectra overlay plot
  - Predicted vs. true coefficients scatter/bar plot
  - Residual norms histogram

### Dataset
- **Built-in:** `_synthetic_mixtures()` function in the script
  - Creates: 2 pure components (Gaussian-like), 6 mixtures (0–100% fractions)
  - Wavenumber grid: 600–1800 cm⁻¹ (120 points)
  - Noise level: σ=0.01

### Teaching Narrative
```
Step 1: Generate pure-component spectra (C1, C2)
Step 2: Create mixture spectra with known fractions [0%, 25%, 50%, ..., 100%]
Step 3: Add realistic noise
Step 4: Apply NNLS unmixing
Step 5: Compare predicted fractions to true values
Step 6: Evaluate reconstruction error and coefficient accuracy
```

### Enhancement Needed
1. Add visualization code (pure spectra plot, coefficient comparison plot)
2. Add expected output summary (e.g., RMSE of predictions)
3. Document limitations (e.g., orthogonality of pure spectra, noise sensitivity)

---

## 4. Hyperspectral Mapping (Spatial Analysis & ROI Segmentation)

**Teaching Goal:** Demonstrate hyperspectral image processing: cube segmentation, ROI extraction, spatial visualization.

### Source Script
- **File:** `examples/hyperspectral_demo.py`
- **Length:** ~51 lines
- **Capabilities Demonstrated:**
  - Loading hyperspectral cube from `HyperspectralDataset`
  - Preprocessing (normalization, smoothing)
  - K-means segmentation to identify regions of interest (ROIs)
  - Extracting mean spectrum per ROI
  - Running RQ engine on ROI-level data
  - Text report generation

### Existing Figures
- `docs/assets/figures/hsi_label_map.png` (0 bytes, needs regeneration)
- `docs/assets/figures/roi_spectra.png` (0 bytes, needs regeneration)

### Dataset
- **Primary:** `examples/data/hsi_synthetic.npz`
  - Shape: (5 spatial_y, 4 spatial_x, 3 wavenumber_bins) — synthetic for fast demo
  - Minimal size for quick execution; can scale to realistic 100×100×1000
- **Built-in:** Could generate synthetic 3D cube

### Teaching Narrative
```
Step 1: Load hyperspectral cube (y, x, wn dimensions)
Step 2: Preprocess all pixels (normalization, smoothing)
Step 3: Segment image into k clusters (e.g., k=2: food vs. background)
Step 4: For each segment, extract mean spectrum (ROI)
Step 5: Convert ROI spectra to peak table (peak picking)
Step 6: Run RQ analysis on aggregated ROI data
Step 7: Generate spatial label map (colored segments)
Step 8: Report quality metrics per ROI
```

### Enhancement Needed
1. Generate and save `hsi_label_map.png` (segmentation map with color legend)
2. Generate and save `roi_spectra.png` (overlay of per-ROI mean spectra)
3. Add expected output: "Segmentation found N ROIs with mean quality scores Q1, Q2, ..."
4. Document cube shape and preprocessing parameters

---

## 5. End-to-End Protocol Run (Unified API)

**Teaching Goal:** Demonstrate the Phase 1 unified `FoodSpec` class with chainable API for complete workflow.

### Source Script
- **File:** `examples/phase1_quickstart.py`
- **Length:** ~139 lines
- **Capabilities Demonstrated:**
  - Creating synthetic spectroscopy dataset
  - Initializing `FoodSpec` class with metadata
  - Chainable API: QC → preprocessing → train → export
  - Model training (classifier + regressor options)
  - Cross-validation and metrics reporting
  - Saving trained model to disk

### Existing Figures
- `docs/assets/figures/architecture_flow.png` (0 bytes, needs regeneration)
- `docs/assets/figures/cv_boxplot.png` (0 bytes, needs regeneration)

### Dataset
- **Built-in:** Synthetic in script
  - 30 samples, 3 oil types, 200 wavenumbers
  - Raman modality (500–2000 cm⁻¹)
  - Metadata: sample_id, oil_type, batch

### Teaching Narrative
```
Step 1: Understand Phase 1 unified API design philosophy
Step 2: Create synthetic Raman data for 3 oil types
Step 3: Initialize FoodSpec with modality and kind
Step 4: Chain QC step (outlier detection)
Step 5: Chain preprocessing (baseline, smoothing, normalization)
Step 6: Chain model training (classifier) with cross-validation
Step 7: Chain predictions on test set
Step 8: Chain model export and metrics reporting
Step 9: Understand error handling and reproducibility (seed)
```

### Enhancement Needed
1. Generate `architecture_flow.png` showing chain diagram (QC → Preprocess → Train → Export)
2. Generate `cv_boxplot.png` showing cross-validation fold performance
3. Update docstring to clarify "Phase 1" vs. "Phase 2" API tiers
4. Add expected output metrics (accuracies per fold, final test accuracy)

---

## Summary Table

| Example | Script | Lines | Key Capability | Dataset | Primary Figures | Status |
|---------|--------|-------|-----------------|---------|-----------------|--------|
| **1. Oil Auth** | `oil_authentication_quickstart.py` | 35 | Classification + PCA | `oil_synthetic.csv` or `load_example_oils()` | `oil_confusion.png`, `pca_scores.png` | 🟡 Script OK, figures regenerate |
| **2. Heating** | `heating_quality_quickstart.py` | 35 | Time-series trend | `synthetic_heating_dataset()` | `heating_ratio_vs_time.png` | 🟡 Script OK, figure exists, needs polish |
| **3. Mixture** | `mixture_analysis_quickstart.py` | 50 | NNLS unmixing | Built-in `_synthetic_mixtures()` | Create new plots | 🔴 Script OK, needs visualization |
| **4. HSI** | `hyperspectral_demo.py` | 51 | Segmentation + ROI | `hsi_synthetic.npz` | `hsi_label_map.png`, `roi_spectra.png` | 🔴 Script OK, figures missing |
| **5. Protocol** | `phase1_quickstart.py` | 139 | Unified API chain | Built-in synthetic | `architecture_flow.png`, `cv_boxplot.png` | 🔴 Script OK, figures missing |

---

## Next Steps (No Code Changes Yet)

1. **Validate** each script runs without errors
2. **Generate** missing figures from scripts
3. **Document** expected outputs and metrics for each example
4. **Create** teaching notebooks (`.ipynb`) wrapping each script with narrative + markdown cells
5. **Link** figures into tutorial docs with captions and interpretation guides
6. **Test** cross-references and backward compatibility
7. **Establish** consistency: all 5 examples use same naming convention, file structure, output folder

---

## Teaching Resource Map

```
Beginner Path:
  1. Oil Authentication (classification basics)
  2. Heating Stability (time-series intro)
  
Intermediate Path:
  3. Mixture Analysis (quantification)
  4. Hyperspectral Mapping (spatial data)
  
Advanced/Capstone:
  5. Protocol End-to-End (unified API mastery)
```

Each example should be **runnable in < 2 minutes**, produce **publication-quality figures**, and **teach a distinct FoodSpec concept**.

