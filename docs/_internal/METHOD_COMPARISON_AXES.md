# Method Comparison Axes for Food Spectroscopy

**Version:** 1.0  
**Date:** January 6, 2026  
**Purpose:** Standardized framework for comparing preprocessing, chemometric, statistical, and QC methods  
**Scope:** Applies to all 142 methods in FoodSpec inventory

---

## Comparison Framework: 12 Axes

### Axis 1: Data Requirements (Sample Size)

**Definition:**  
Minimum number of samples needed for the method to produce reliable, statistically valid results.

**Scale:**
- **Low** (n < 30): Works with very small datasets; suitable for pilot studies or rare samples
- **Medium** (30 ≤ n < 100): Requires moderate sample sizes; typical for exploratory studies
- **High** (n ≥ 100): Needs large datasets; required for robust modeling

**Why it matters:**  
Food spectroscopy studies often face sample constraints (cost, availability, destructive testing). Methods requiring fewer samples enable feasibility studies and rare sample analysis. Mismatched sample size can lead to overfitting, unstable parameters, or invalid statistical tests.

**Example (FoodSpec context):**  
- **Low**: Paired t-test on heating quality (n=15 time points from 3 oils) ✅ Valid
- **Medium**: PLS-DA for oil authentication (n=50, 5 classes) → Minimum 10/class
- **High**: Random Forest with 1000 spectral features (n=200) → Rule of thumb: n > 10×features for stable importance

---

### Axis 2: Sensitivity to Baseline Drift

**Definition:**  
Degree to which uncorrected baseline variations (fluorescence, slopes, curvature) distort method outputs or invalidate results.

**Scale:**
- **Low**: Method inherently insensitive (e.g., derivatives suppress baseline) or operates on baseline-corrected data
- **Medium**: Baseline affects results moderately; can tolerate mild drift
- **High**: Method critically assumes flat baseline; any drift causes bias

**Why it matters:**  
Raman fluorescence and FTIR-ATR contact variations introduce baseline artifacts. Methods highly sensitive to baseline require mandatory preprocessing, while robust methods reduce workflow complexity.

**Example (FoodSpec context):**  
- **Low**: 2nd derivative + PCA (derivative removes baseline) → Robust to fluorescence in Raman oil spectra
- **Medium**: Vector normalization (removes scale but not slope) → Tolerates mild ATR contact variation
- **High**: Raw peak height ratios (1655/1444 cm⁻¹) → Fluorescent background in olive oil biases numerator more than denominator, invalidating ratio

---

### Axis 3: Sensitivity to Scattering Effects

**Definition:**  
Impact of multiplicative scatter (particle size, path length, ATR pressure) on method performance without scatter correction.

**Scale:**
- **Low**: Method immune to multiplicative scaling (e.g., SNV pre-applied, ratio-based)
- **Medium**: Scatter introduces noise but doesn't invalidate method
- **High**: Scatter directly confounds signal of interest; MSC/SNV mandatory

**Why it matters:**  
Powdered samples (spices, chips) and ATR measurements exhibit variable scatter. Scatter-sensitive methods require preprocessing or fail silently by learning instrument artifacts instead of chemistry.

**Example (FoodSpec context):**  
- **Low**: Peak area ratios (1655/2920 cm⁻¹) after SNV → Scatter cancels in ratio
- **Medium**: PCA scores (captures scatter as PC1/PC2) → Can cluster by particle size instead of composition
- **High**: Raw spectrum intensity → Chip thickness variation (50–200 µm) dominates signal; classifier learns thickness, not oil type

---

### Axis 4: Robustness to Instrument Variation

**Definition:**  
Ability to produce consistent results across different instruments, sessions, or operators without recalibration.

**Scale:**
- **Low**: Method output changes significantly across instruments; requires calibration transfer
- **Medium**: Minor batch effects tolerable; periodic QC checks sufficient
- **High**: Method inherently instrument-independent or uses normalized/ratio features

**Why it matters:**  
Multi-site studies, instrument upgrades, and long-term monitoring require methods that don't degrade across batches. Low robustness methods demand expensive harmonization protocols.

**Example (FoodSpec context):**  
- **Low**: PLS regression coefficients trained on Instrument A → RMSE doubles on Instrument B (different Raman laser power)
- **Medium**: PCA loadings shift slightly → 95% overlap in scores between instruments after standardization
- **High**: RQ ratio (1655/2920 cm⁻¹) thresholds (>1.5 = adulterated) → Same threshold works across 3 labs using different FT-IR spectrometers

---

### Axis 5: Interpretability

**Definition:**  
Ease of explaining method mechanics and results to non-expert stakeholders (food scientists, regulators, industry).

**Scale:**
- **Low**: Black-box model; outputs not traceable to chemical features (e.g., deep neural nets)
- **Medium**: Coefficients/loadings available but require chemometric expertise to interpret
- **High**: Direct chemical meaning; results map to specific peaks/functional groups

**Why it matters:**  
Regulatory submissions, publication, and industry adoption require explainability. High interpretability enables hypothesis generation and builds trust. Low interpretability limits method to screening applications.

**Example (FoodSpec context):**  
- **Low**: Random Forest with 1000 features → "Model predicts adulteration" (hard to explain *why*)
- **Medium**: PLS-DA loadings → "Peaks at 1655, 1745 cm⁻¹ differentiate olive from sunflower" (requires spectroscopy knowledge)
- **High**: RQ ratio 1655/2920 → "Unsaturation index < 1.2 flags saturated fat addition" (direct chemical interpretation)

---

### Axis 6: Computational Cost

**Definition:**  
Time and memory required to train/fit method and make predictions on typical FoodSpec datasets (100–1000 spectra, 500–2000 features).

**Scale:**
- **Low**: < 1 second on laptop CPU (simple operations, closed-form solutions)
- **Medium**: 1–60 seconds (iterative algorithms, moderate complexity)
- **High**: > 60 seconds or requires GPU (deep learning, large-scale optimization)

**Why it matters:**  
Real-time QC applications, large HSI datasets, and interactive exploration require fast methods. High-cost methods limit prototyping speed and deployment scenarios.

**Example (FoodSpec context):**  
- **Low**: Baseline ALS correction (500 spectra, 1000 points) → 0.5s → Real-time preprocessing in QC pipeline
- **Medium**: PLS-DA cross-validation (200 spectra, 5 folds) → 15s → Acceptable for offline analysis
- **High**: MCR-ALS mixture resolution (HSI cube: 100×100×1000) → 5 minutes → Batch processing only

---

### Axis 7: Assumptions (Statistical/Mathematical)

**Definition:**  
Critical assumptions that, if violated, invalidate method results or lead to misleading conclusions.

**Scale:**
- **Low**: Few or no assumptions; nonparametric or distribution-free
- **Medium**: Moderate assumptions (linearity, homoscedasticity) that are testable/robust
- **High**: Strict assumptions (normality, independence, equal variance) that rarely hold in real data

**Why it matters:**  
Violated assumptions cause inflated Type I errors, biased estimates, and unreliable conclusions. High-assumption methods require diagnostic checks; low-assumption methods offer safety.

**Example (FoodSpec context):**  
- **Low**: Mann-Whitney U test on peak ratios → No normality assumption; valid even for n=5/group with skew
- **Medium**: PLS regression → Assumes linear relationship between spectra and concentration; fails for saturation effects
- **High**: ANOVA on heating degradation → Assumes normality + equal variance across 5 time points; violated by exponential decay trajectories

---

### Axis 8: Signal Quality Requirements (SNR)

**Definition:**  
Minimum signal-to-noise ratio needed for method to function reliably without excessive false positives/negatives.

**Scale:**
- **Low**: SNR > 50 required; fails on noisy data (e.g., unsmoothed portable Raman)
- **Medium**: SNR 10–50 acceptable; moderate noise tolerated
- **High**: SNR < 10 tolerated; robust to very noisy data (e.g., bootstrapped metrics)

**Why it matters:**  
Portable instruments, rapid acquisition, and challenging matrices (fluorescent, opaque) produce low-SNR spectra. Methods requiring high SNR force longer acquisition times or restrict deployment.

**Example (FoodSpec context):**  
- **Low**: 2nd derivative peak detection → SNR < 20 produces spurious peaks; fails on 1s handheld Raman scans
- **Medium**: PCA (first 5 PCs) → SNR = 30 sufficient; noise contributes to higher PCs only
- **High**: Median-smoothed peak area ratios → SNR = 5 acceptable for QC pass/fail; robust to shot noise in online monitoring

---

### Axis 9: Preprocessing Dependency

**Definition:**  
Extent to which method requires specific preprocessing steps to produce valid results; sensitivity to preprocessing order/parameters.

**Scale:**
- **Low**: Works on raw or minimally processed data; insensitive to preprocessing choices
- **Medium**: Benefits from standard preprocessing but tolerates variations
- **High**: Critically depends on exact preprocessing pipeline; results change dramatically with parameter tuning

**Why it matters:**  
High dependency increases workflow complexity, introduces tuning parameters (overfitting risk), and reduces reproducibility across labs. Low dependency simplifies deployment.

**Example (FoodSpec context):**  
- **Low**: PCA on SNV-normalized spectra → Robust to choice of baseline method (ALS vs polynomial)
- **Medium**: Peak detection → Works better after smoothing (window=5 vs 9 changes peak count by ~10%)
- **High**: 2nd derivative SVM → Results depend on Savitzky-Golay window (5 vs 15 changes accuracy by 20%); parameter must be optimized

---

### Axis 10: Failure Modes / Misuse Risk

**Definition:**  
Likelihood and severity of silent failures (wrong results without warning) when method is misapplied or assumptions violated.

**Scale:**
- **Low**: Method fails loudly (errors, NaN, divergence) or has narrow failure scenarios
- **Medium**: Failure possible but detectable via diagnostics (residuals, metrics)
- **High**: Silently produces plausible but wrong results; user unaware of problem

**Why it matters:**  
High-risk methods can lead to false discoveries, regulatory issues, or product recalls if misapplied. Low-risk methods offer safety for non-experts.

**Example (FoodSpec context):**  
- **Low**: PCA with negative eigenvalues → Raises error; forces user to fix data issues
- **Medium**: PLS overfitting (n=50, 40 components) → Low Q² signals problem; user should check diagnostics
- **High**: Classification on class-imbalanced data (95% authentic, 5% adulterated) → 95% accuracy looks good but model predicts "authentic" for everything; misses all adulterations

---

### Axis 11: Feature Type Compatibility

**Definition:**  
Types of input features the method can handle effectively (full spectra, peak positions, areas, ratios, PC scores).

**Scale:**
- **Narrow**: Single feature type only (e.g., ratios-only, full-spectra-only)
- **Moderate**: Works with 2-3 feature types with minor adaptations
- **Broad**: Handles any feature representation; feature-agnostic

**Why it matters:**  
Workflow flexibility depends on feature compatibility. Narrow methods force specific feature engineering; broad methods enable exploratory analysis.

**Example (FoodSpec context):**  
- **Narrow**: RQ Engine → Requires peak definitions (numerator/denominator wavenumbers); cannot use full spectra
- **Moderate**: PLS-DA → Designed for full spectra but works with PCA scores or band areas as input
- **Broad**: t-test → Works on any numeric feature (peak heights, ratios, PC scores, concentrations)

---

### Axis 12: Handling of Nonlinearity

**Definition:**  
Ability to model nonlinear relationships between predictors and response without manual feature engineering.

**Scale:**
- **Linear Only**: Assumes/requires linear relationships; fails on curved or threshold effects
- **Weakly Nonlinear**: Handles mild curvature via transformations or local approximations
- **Strongly Nonlinear**: Natively models arbitrary nonlinear mappings

**Why it matters:**  
Food chemistry involves nonlinear processes (oxidation kinetics, concentration quenching, Beer's Law deviations). Linear methods require transformations (log, sqrt) or fail; nonlinear methods capture effects directly.

**Example (FoodSpec context):**  
- **Linear Only**: Linear regression on concentration → Fails for absorbance saturation at high concentrations (Beer's Law breakdown); requires log-transform
- **Weakly Nonlinear**: PLS with quadratic terms → Captures parabolic heating trajectories (peak height vs time²)
- **Strongly Nonlinear**: Random Forest → Directly learns threshold effects (e.g., "RQ ratio > 1.5 *and* peak 1745 > 0.2 → adulterated"); no feature engineering needed

---

## Usage Guidelines

### Axis Selection for Comparison Tables

**For preprocessing methods**, prioritize:
- Sensitivity to Baseline Drift (Axis 2)
- Sensitivity to Scattering (Axis 3)
- Computational Cost (Axis 6)
- Failure Modes (Axis 10)

**For chemometric models**, prioritize:
- Data Requirements (Axis 1)
- Interpretability (Axis 5)
- Assumptions (Axis 7)
- Handling of Nonlinearity (Axis 12)

**For statistical tests**, prioritize:
- Data Requirements (Axis 1)
- Assumptions (Axis 7)
- Failure Modes (Axis 10)
- Feature Type Compatibility (Axis 11)

**For QC methods**, prioritize:
- Robustness to Instrument Variation (Axis 4)
- Signal Quality Requirements (Axis 8)
- Preprocessing Dependency (Axis 9)
- Computational Cost (Axis 6)

### Rating Scale Standardization

For matrix population, use this mapping:

| Numeric | Symbol | Description |
|---------|--------|-------------|
| 1 | ● | Low/Poor/Narrow |
| 2 | ◐ | Medium/Moderate |
| 3 | ○ | High/Good/Broad |

### Qualitative Descriptors by Axis

| Axis | Low (●) | Medium (◐) | High (○) |
|------|---------|-----------|----------|
| 1. Data Requirements | n < 30 | 30 ≤ n < 100 | n ≥ 100 |
| 2. Sensitivity to Baseline | Immune/corrected | Tolerates mild drift | Critically sensitive |
| 3. Sensitivity to Scattering | Immune/normalized | Introduces noise | Directly confounded |
| 4. Robustness (Instruments) | Requires transfer | Periodic QC needed | Inherently robust |
| 5. Interpretability | Black-box | Requires expertise | Direct chemical meaning |
| 6. Computational Cost | > 60s or GPU | 1-60s | < 1s |
| 7. Assumptions | Strict (normality, etc.) | Moderate (linearity) | Few/none |
| 8. Signal Quality Req. | SNR > 50 | SNR 10-50 | SNR < 10 tolerated |
| 9. Preprocessing Dependency | Critical/exact pipeline | Benefits but flexible | Works on raw data |
| 10. Failure Modes | Silent failures | Detectable via diagnostics | Fails loudly |
| 11. Feature Compatibility | Single type | 2-3 types | Feature-agnostic |
| 12. Nonlinearity Handling | Linear only | Weakly nonlinear | Strongly nonlinear |

---

## Application Examples

### Example 1: Comparing Baseline Methods

| Method | Axis 2 (Self) | Axis 6 (Cost) | Axis 10 (Failure) |
|--------|---------------|---------------|-------------------|
| **ALS** | ○ (removes drift) | ◐ (10s/500 spectra) | ◐ (over-smoothing detectable) |
| **Rubberband** | ○ (removes drift) | ○ (<1s) | ● (silent failure if peaks dense) |
| **Polynomial** | ◐ (mild drift only) | ○ (<1s) | ◐ (overfitting with high degree) |

### Example 2: Comparing Classifiers

| Method | Axis 1 (Data Req.) | Axis 5 (Interpret.) | Axis 12 (Nonlinearity) |
|--------|-------------------|---------------------|------------------------|
| **PLS-DA** | ◐ (n=50) | ◐ (loadings) | ● (linear only) |
| **Random Forest** | ○ (n=100+) | ● (black-box) | ○ (strongly nonlinear) |
| **Logistic Reg** | ● (n=20) | ○ (coefficients) | ● (linear only) |

### Example 3: Comparing Statistical Tests

| Method | Axis 1 (Data Req.) | Axis 7 (Assumptions) | Axis 10 (Failure) |
|--------|-------------------|----------------------|-------------------|
| **t-test** | ● (n=5/group) | ● (normality + equal var) | ● (inflated α if violated) |
| **Mann-Whitney U** | ● (n=5/group) | ○ (distribution-free) | ○ (robust to violations) |
| **ANOVA** | ◐ (n=10/group) | ● (normality + homoscedasticity) | ● (silent if violated) |

---

## Cross-Axis Considerations

### Trade-offs

1. **Interpretability vs Nonlinearity** (Axes 5 & 12):  
   High interpretability methods (ratios, linear models) struggle with nonlinear chemistry. Nonlinear models (RF, SVM-RBF) sacrifice explainability.

2. **Data Requirements vs Robustness** (Axes 1 & 4):  
   Large-n methods can learn instrument-specific artifacts. Small-n methods force simple, generalizable features.

3. **Computational Cost vs Nonlinearity** (Axes 6 & 12):  
   Nonlinear methods (deep learning, MCR-ALS) are computationally expensive. Fast methods (linear regression, t-tests) are limited to linear regimes.

### Method Selection Decision Tree

```
Q1: Is n < 30?
├─ YES → Use Low-data methods (Axis 1: ●)
│         → Avoid Random Forest, deep learning
└─ NO  → Proceed to Q2

Q2: Is interpretability critical (regulatory, publication)?
├─ YES → Use High-interpret. methods (Axis 5: ○)
│         → Ratios, PLS, linear models
└─ NO  → Proceed to Q3

Q3: Are relationships nonlinear (thresholds, kinetics)?
├─ YES → Use Nonlinear methods (Axis 12: ○)
│         → Random Forest, SVM-RBF, polynomials
└─ NO  → Use Linear methods
          → PLS, linear regression, t-tests
```

---

## Validation & Maintenance

### Axis Review Process
- **Frequency**: Annually or when adding major method categories
- **Reviewers**: Core team + domain experts (food chemists, statisticians)
- **Update triggers**: New method types, user feedback on comparison utility

### Method Rating Process
1. Assign 2 independent raters per method
2. Resolve disagreements via discussion + test cases
3. Document edge cases in method comparison matrix
4. Validate ratings via synthetic/benchmark datasets

---

**Framework Maintained By:** FoodSpec Core Team  
**Contributors:** Food Scientists, Chemometricians, Statisticians  
**Next Review:** Q1 2027 or post-v2.0 release
