# Canonical Method Selection for Comparison Matrix

**Purpose:** Minimal representative set of methods for `docs/reference/method_comparison.md`  
**Scope:** 18 methods across 4 families (from 142 total inventory)  
**Selection Criteria:** Default (recommended first choice), Alternative (valid for specific cases), Discouraged (edge-case only)

---

## Preprocessing Methods (5 methods)

| Family | Method | Status | One-line Rationale | Method Page | Example |
|--------|--------|--------|-------------------|-------------|---------|
| Preprocessing | **ALS Baseline** | Default | Flexible, handles moderate-strong curvature, parameter-tunable | [baseline_correction.md](../methods/preprocessing/baseline_correction.md) | [01_oil_authentication.md](../examples/01_oil_authentication.md) |
| Preprocessing | **Rubberband Baseline** | Alternative | Fast, parameter-free, but fails on dense peaks | [baseline_correction.md](../methods/preprocessing/baseline_correction.md) | N/A |
| Preprocessing | **Polynomial Baseline** | Discouraged | Overfits with high degree, unsuitable for complex fluorescence | [baseline_correction.md](../methods/preprocessing/baseline_correction.md) | N/A |
| Preprocessing | **SNV Normalization** | Default | Standard for scatter correction, removes multiplicative effects | [normalization_smoothing.md](../methods/preprocessing/normalization_smoothing.md) | [01_oil_authentication.md](../examples/01_oil_authentication.md) |
| Preprocessing | **Savitzky-Golay Smoothing** | Default | Preserves peak shape while reducing noise, suitable before derivatives | [normalization_smoothing.md](../methods/preprocessing/normalization_smoothing.md) | [02_heating_quality_monitoring.md](../examples/02_heating_quality_monitoring.md) |

---

## Chemometrics Models (5 methods)

| Family | Method | Status | One-line Rationale | Method Page | Example |
|--------|--------|--------|-------------------|-------------|---------|
| Chemometrics | **PLS-DA** | Default | Standard for supervised classification, interpretable loadings, handles correlated features | [classification_regression.md](../methods/chemometrics/classification_regression.md) | [01_oil_authentication.md](../examples/01_oil_authentication.md) |
| Chemometrics | **PCA** | Default | Exploratory analysis, dimensionality reduction, visualize class separation | [pca_and_dimensionality_reduction.md](../methods/chemometrics/pca_and_dimensionality_reduction.md) | [01_oil_authentication.md](../examples/01_oil_authentication.md) |
| Chemometrics | **Random Forest** | Alternative | Nonlinear, robust to noise, handles interactions but less interpretable | [classification_regression.md](../methods/chemometrics/classification_regression.md) | N/A |
| Chemometrics | **PLS Regression** | Default | Calibration and property prediction, standard for concentration modeling | [classification_regression.md](../methods/chemometrics/classification_regression.md) | [03_mixture_analysis.md](../examples/03_mixture_analysis.md) |
| Chemometrics | **Deep Learning (Conv1D)** | Discouraged | Requires n>500, black-box, overkill for typical food spec datasets | [advanced_deep_learning.md](../methods/chemometrics/advanced_deep_learning.md) | N/A |

---

## Validation Strategies (4 methods)

| Family | Method | Status | One-line Rationale | Method Page | Example |
|--------|--------|--------|-------------------|-------------|---------|
| Validation | **Stratified k-Fold CV** | Default | Maintains class balance, provides CI on metrics, standard for classification | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | [01_oil_authentication.md](../examples/01_oil_authentication.md) |
| Validation | **Nested CV** | Alternative | Unbiased hyperparameter selection + evaluation, use when tuning models | [advanced_validation_strategies.md](../methods/validation/advanced_validation_strategies.md) | N/A |
| Validation | **Leave-One-Out CV** | Alternative | Maximum training data usage for small n (<30), computationally expensive | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | N/A |
| Validation | **Single Train/Test Split** | Discouraged | No CI, unstable metrics, only acceptable for very large n or final holdout | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | N/A |

---

## Statistical Tests (6 methods)

| Family | Method | Status | One-line Rationale | Method Page | Example |
|--------|--------|--------|-------------------|-------------|---------|
| Statistics | **Independent t-test** | Default | Compare two groups, parametric, requires normality + equal variance | [t_tests_effect_sizes_and_power.md](../methods/statistics/t_tests_effect_sizes_and_power.md) | [02_heating_quality_monitoring.md](../examples/02_heating_quality_monitoring.md) |
| Statistics | **One-Way ANOVA** | Default | Compare 3+ groups, parametric, post-hoc tests required for pairwise comparisons | [anova_and_manova.md](../methods/statistics/anova_and_manova.md) | N/A |
| Statistics | **Mann-Whitney U** | Alternative | Nonparametric t-test alternative, robust to non-normality and outliers | [nonparametric_methods_and_robustness.md](../methods/statistics/nonparametric_methods_and_robustness.md) | N/A |
| Statistics | **Kruskal-Wallis** | Alternative | Nonparametric ANOVA alternative, use when normality violated | [nonparametric_methods_and_robustness.md](../methods/statistics/nonparametric_methods_and_robustness.md) | N/A |
| Statistics | **Tukey HSD** | Default | Post-hoc for ANOVA, assumes equal variances, family-wise error control | [anova_and_manova.md](../methods/statistics/anova_and_manova.md) | N/A |
| Statistics | **Games-Howell** | Alternative | Post-hoc for unequal variances, more robust than Tukey but computationally intensive | [anova_and_manova.md](../methods/statistics/anova_and_manova.md) | N/A |

---

## Selection Rationale

### Preprocessing (5/37 methods)
**Included:**
- Baseline: ALS (default), Rubberband (fast alt), Polynomial (cautionary example)
- Normalization: SNV (default, most common)
- Smoothing: Savitzky-Golay (default, peak-preserving)

**Excluded:**
- MSC (similar to SNV, redundant for comparison)
- Derivatives (specialized use, not core preprocessing)
- Atmospheric/ATR correction (instrument-specific)
- Peak detection (feature extraction, not preprocessing)

### Chemometrics (5/25 methods)
**Included:**
- Classification: PLS-DA (default), Random Forest (nonlinear alt)
- Dimensionality: PCA (default exploratory)
- Regression: PLS-R (default calibration)
- Deep Learning: Conv1D (cautionary example for small n)

**Excluded:**
- SVM (similar to RF, redundant nonlinear classifier)
- Logistic Regression (linear like PLS-DA, less common in chemometrics)
- kNN (toy baseline, not production)
- NNLS/MCR-ALS (mixture-specific, not general classification/regression)
- SIMCA (niche, class modeling vs classification)

### Validation (4/18 methods)
**Included:**
- Stratified k-Fold (default, most common)
- Nested CV (hyperparameter tuning best practice)
- LOO (small-n alternative)
- Train/Test Split (cautionary anti-pattern)

**Excluded:**
- Permutation tests (robustness check, not primary validation)
- Bootstrap (uncertainty quantification, not primary validation)
- Conformal prediction (advanced, niche)
- Metrics (ROC, confusion matrix) are outputs, not strategies

### Statistics (6/23 methods)
**Included:**
- Parametric: t-test, ANOVA (defaults for 2 and 3+ groups)
- Nonparametric: Mann-Whitney U, Kruskal-Wallis (robust alternatives)
- Post-hoc: Tukey HSD (default), Games-Howell (unequal variance alt)

**Excluded:**
- Paired t-test (specific case of t-test, not separate comparison)
- MANOVA (multivariate extension, less common)
- Wilcoxon/Friedman (paired/repeated measures, specific designs)
- Cohen's d (effect size, not hypothesis test)
- Correlations (association, not group comparison)

---

## Summary Table

| Family | Total Methods | Selected | Coverage |
|--------|--------------|----------|----------|
| Preprocessing | 37 | 5 | 13.5% (core workflow) |
| Chemometrics | 25 | 5 | 20.0% (canonical models) |
| Validation | 18 | 4 | 22.2% (essential strategies) |
| Statistics | 23 | 6 | 26.1% (common tests) |
| **TOTAL** | **103** | **20** | **19.4%** |

---

## Next Steps for method_comparison.md

Using these 20 methods:

1. **Create comparison matrix** using axes from METHOD_COMPARISON_AXES.md
2. **Populate ratings** (●◐○) for each method × axis intersection
3. **Add decision flowchart** (e.g., "baseline drift present? → Use ALS or Rubberband")
4. **Include code snippets** showing API calls for each method
5. **Link to examples** where methods are demonstrated

**Target page size:** 1500-2000 words (vs 73 words placeholder currently)
