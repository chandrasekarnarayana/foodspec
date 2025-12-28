# Metric Interpretation & Significance Tables

!!! info "Context Block"
    **Purpose:** Comprehensive reference for interpreting classification, regression, and statistical metrics with significance thresholds, context-dependent guidance, and quality criteria.
    
    **Audience:** Analysts, reviewers, quality control labs, regulatory auditors
    
    **Prerequisites:** Basic statistics knowledge (means, standard deviations, p-values)
    
    **Related:** [Metrics & Evaluation](../reference/metrics_reference.md) | [Statistical Power](../protocols/statistical_power_and_limits.md) | [Hypothesis Testing](../methods/statistics/hypothesis_testing_in_food_spectroscopy.md)

---

## Overview

Metric values alone are insufficient — **context determines significance**. A p-value of 0.001 means nothing if the effect size is negligible. An AUC of 0.95 is excellent for balanced data but may hide poor minority class performance. This page provides:

1. **Threshold tables** for classification, regression, and effect sizes
2. **Context-dependent interpretation** (sample size, class balance, domain)
3. **Significance tests** to compare models or validate improvements
4. **Red flags** indicating when metrics mislead

---

## Classification Metrics

### Performance Thresholds

| Metric | Range | Poor | Fair | Good | Excellent | Recommended Significance Test |
|--------|-------|------|------|------|-----------|-------------------------------|
| **Accuracy** | 0–1 | <0.7 | 0.7–0.8 | 0.8–0.9 | >0.9 | McNemar's test (paired), permutation test |
| **F1 Score** | 0–1 | <0.6 | 0.6–0.75 | 0.75–0.85 | >0.85 | Bootstrap CI on F1 |
| **AUC-ROC** | 0–1 | <0.7 | 0.7–0.8 | 0.8–0.9 | >0.9 | DeLong's test (paired AUCs) |
| **Precision** | 0–1 | <0.7 | 0.7–0.85 | 0.85–0.95 | >0.95 | Binomial confidence interval |
| **Recall (Sensitivity)** | 0–1 | <0.6 | 0.6–0.8 | 0.8–0.9 | >0.9 | Binomial confidence interval |
| **Specificity** | 0–1 | <0.6 | 0.6–0.8 | 0.8–0.9 | >0.9 | Binomial confidence interval |
| **Balanced Accuracy** | 0–1 | <0.65 | 0.65–0.75 | 0.75–0.85 | >0.85 | Permutation test |
| **Matthews Correlation (MCC)** | -1–1 | <0.3 | 0.3–0.5 | 0.5–0.7 | >0.7 | Bootstrap CI on MCC |
| **Cohen's Kappa** | -1–1 | <0.4 | 0.4–0.6 | 0.6–0.8 | >0.8 | Bootstrap CI on Kappa |

### Context-Dependent Interpretation

#### Imbalanced Data (Class Ratio > 3:1)
- **DO NOT use accuracy** (misleading when minority class is important)
- **Prefer:** F1, Balanced Accuracy, AUC, MCC, Kappa
- **Report:** Per-class precision/recall + confusion matrix
- **Example:** 95% accuracy detecting adulteration (1% prevalence) = 0% recall on adulterants

#### Rare Event Detection (Prevalence < 5%)
- **Prioritize recall** over precision (minimize false negatives)
- **Accept lower precision** (tolerate false alarms)
- **Use:** Cost-sensitive learning (higher penalty for false negatives)
- **Example:** Food safety (miss no contaminated samples, even if many false positives)

#### Screening vs. Confirmation
- **Screening:** High recall (>0.95), lower precision acceptable
- **Confirmation:** High precision (>0.95), lower recall acceptable
- **Two-stage workflow:** Screen broadly → confirm positives with reference method

#### Small Sample Size (n < 100)
- **Wide confidence intervals** (report 95% CI, not just point estimate)
- **Beware overfitting:** Cross-validated metrics < training metrics
- **Use:** Nested cross-validation, repeated CV (5×2 or 10×5)
- **Red flag:** Training accuracy = 1.0, test accuracy < 0.8

---

## Regression Metrics

### Performance Thresholds

| Metric | Range | Poor | Fair | Good | Excellent | Significance Test |
|--------|-------|------|------|------|-----------|-------------------|
| **R² (Coefficient of Determination)** | 0–1 | <0.5 | 0.5–0.7 | 0.7–0.85 | >0.85 | F-test (nested models), permutation test |
| **Adjusted R²** | 0–1 | <0.4 | 0.4–0.6 | 0.6–0.8 | >0.8 | Compare with R² (overfitting check) |
| **RMSE (Root Mean Squared Error)** | 0–∞ | >2σ | 1–2σ | 0.5–1σ | <0.5σ | Bootstrap CI; compare to baseline RMSE |
| **MAE (Mean Absolute Error)** | 0–∞ | >1.5σ | 0.75–1.5σ | 0.4–0.75σ | <0.4σ | Bootstrap CI |
| **MAPE (Mean Absolute % Error)** | 0–∞ | >20% | 10–20% | 5–10% | <5% | Bootstrap CI |
| **Q² (Cross-validated R²)** | -∞–1 | <0.4 | 0.4–0.6 | 0.6–0.8 | >0.8 | Compare with R² (overfitting = R² >> Q²) |

**σ = standard deviation of target variable (y)**

### Context-Dependent Interpretation

#### Small vs. Large Datasets
- **Small (n < 50):** R² overoptimistic → report cross-validated Q²
- **Large (n > 1000):** Tiny R² improvements statistically significant but not practically meaningful
- **Check effect size:** RMSE reduction > 10% of σ for practical significance

#### Heteroscedastic Data (Error Variance ≠ Constant)
- **RMSE penalizes large errors** (sensitive to outliers)
- **MAE more robust** to heteroscedasticity
- **Use:** Weighted least squares or quantile regression

#### Multicollinearity (VIF > 5)
- **R² inflated** (overfitting to correlated predictors)
- **Report VIF** (Variance Inflation Factor) for all predictors
- **Use:** Ridge regression (L2 penalty) or PLS (latent variables)

#### Overfitting Detection
| Symptom | Diagnosis | Remedy |
|---------|-----------|--------|
| R² >> Q² (gap > 0.2) | Overfitting to training data | Increase λ penalty, reduce features, add data |
| Adj. R² << R² (gap > 0.1) | Too many features for sample size | Remove low-importance features |
| Training R² = 1.0 | Perfect fit (memorization) | Check for data leakage or duplicate samples |

---

## Statistical Significance & p-values

### p-value Interpretation

| p-value | Interpretation | Caveat |
|---------|----------------|--------|
| **p < 0.001** | Highly statistically significant | Check effect size (small effect + large n → low p) |
| **p < 0.01** | Statistically significant | Not proof of causation; check assumptions |
| **p < 0.05** | Marginally significant | 5% false positive rate (α); correct for multiple tests |
| **p > 0.05** | Not statistically significant | **≠ proof of no effect** (may be underpowered) |

### Critical Warnings

!!! danger "When p-values Mislead"
    1. **Small effect + large n** → p < 0.001 but practically irrelevant (e.g., AUC difference = 0.51 vs. 0.50)
    2. **Large effect + small n** → p > 0.05 but important (e.g., 50% improvement, but n=15)
    3. **Multiple testing** → p < 0.05 expected by chance (5% false positive rate)
    4. **p-hacking** → Trying many tests until p < 0.05 (cherry-picking)

**Solution:** Always report effect size alongside p-value + confidence intervals.

---

## Effect Sizes

### Thresholds & Interpretation

| Effect Size | Cohen's d | η² (ANOVA) | Cramér's V | Interpretation | Practical Significance |
|-------------|-----------|------------|------------|----------------|------------------------|
| **Negligible** | <0.2 | <0.01 | <0.1 | Trivial difference | Unlikely to matter in practice |
| **Small** | 0.2–0.5 | 0.01–0.06 | 0.1–0.3 | Detectable with instruments | May matter in high-precision tasks |
| **Medium** | 0.5–0.8 | 0.06–0.14 | 0.3–0.5 | Moderate difference | Relevant for quality control |
| **Large** | 0.8–1.2 | 0.14–0.20 | 0.5–0.7 | Substantial difference | Clear practical impact |
| **Very Large** | >1.2 | >0.20 | >0.7 | Dominant effect | Major effect; check for artifacts |

### When to Use Which Effect Size

| Scenario | Recommended Effect Size | Reason |
|----------|------------------------|--------|
| Two-group comparison (t-test) | **Cohen's d** | Standardized mean difference (unit-free) |
| ANOVA (3+ groups) | **η² (eta-squared)** | Proportion of variance explained |
| Categorical association (Chi-square) | **Cramér's V** | Normalized for table size |
| Regression | **R²** or **f²** (Cohen's f-squared) | Variance explained by predictor(s) |
| Non-parametric tests | **rank-biserial r** | Rank-based effect size |

### Effect Size + p-value Decision Matrix

| Effect Size | p-value | Interpretation | Action |
|-------------|---------|----------------|--------|
| **Large** | p < 0.05 | Significant + meaningful | **Proceed** with confidence |
| **Large** | p > 0.05 | Likely underpowered | Increase sample size; may still be real |
| **Small** | p < 0.05 | Significant but trivial | **Caution:** Statistically significant ≠ important |
| **Small** | p > 0.05 | Not significant, not meaningful | No evidence of effect |

---

## Sample Size & Power

### Minimum Sample Size Guidelines

| Analysis Type | Minimum n | Preferred n | Rationale |
|---------------|-----------|-------------|-----------|
| **t-test** | 30 per group | 50 per group | Central Limit Theorem (CLT) applies at n≈30 |
| **ANOVA (3 groups)** | 20 per group | 30 per group | Equal group sizes; robust to normality at n≥20 |
| **Linear regression** | 10–15 per predictor | 20 per predictor | Prevent overfitting (Harrell's rule: n/10) |
| **PLS-DA** | 5–10 per latent variable | 10–15 per LV | Spectroscopy rule: 5–10 samples per LV |
| **Cross-validation** | 50 total | 100 total | Stratified 5-fold requires ≥10 per fold |

### Power Analysis (Detecting True Effects)

**Statistical power = Probability of detecting effect when it exists (1 - β)**

| Power | Interpretation | Consequence of Low Power |
|-------|----------------|--------------------------|
| **0.80 (80%)** | Conventional minimum | 20% chance of false negative (Type II error) |
| **0.90 (90%)** | Recommended for critical decisions | 10% chance of false negative |
| **<0.50 (50%)** | Underpowered (coin flip) | High risk of missing real effects |

**Factors affecting power:**

1. **Sample size (n)** ↑ → Power ↑
2. **Effect size** ↑ → Power ↑
3. **Significance level (α)** ↑ → Power ↑ (but more false positives)
4. **Variability (σ)** ↓ → Power ↑

**Power calculation tools:**

- R: `pwr` package
- Python: `statsmodels.stats.power`
- Online: G*Power, GPower calculator

---

## Multiple Testing Correction

### When Multiple Tests Inflate False Positives

Testing 100 hypotheses at α = 0.05 → expect 5 false positives by chance.

**Example:** Testing 1000 wavenumbers for group differences → ~50 false positives at α = 0.05.

### Correction Methods

| Method | Control | Description | Use When |
|--------|---------|-------------|----------|
| **Bonferroni** | FWER | α_corrected = α / m (m = # tests) | Few tests (<20); very conservative |
| **Holm-Bonferroni** | FWER | Step-down Bonferroni (less conservative) | Moderate tests (20–100) |
| **Benjamini-Hochberg (BH)** | FDR | Controls false discovery rate | Many tests (>100); exploratory analysis |
| **Permutation tests** | Exact p-value | Empirical null distribution | Non-parametric; computational cost OK |

**FWER = Family-Wise Error Rate (probability of ≥1 false positive)**  
**FDR = False Discovery Rate (expected proportion of false positives)**

### Bonferroni Example

- α = 0.05, m = 20 tests
- α_corrected = 0.05 / 20 = 0.0025
- **Reject H₀ if p < 0.0025** (not 0.05)

---

## Red Flags: When Metrics Mislead

### Classification Red Flags

| Red Flag | Problem | Solution |
|----------|---------|----------|
| Accuracy = 99%, but 99% majority class | Predicting majority class only | Use balanced accuracy, F1, AUC |
| AUC = 0.95, but precision = 0.1 | Imbalanced data; low positive predictive value | Report precision-recall curve + F1 |
| Training accuracy = 1.0, test = 0.7 | Severe overfitting | Reduce model complexity, add regularization |
| All samples predicted as one class | Model collapsed | Check class weights, data balance, learning rate |

### Regression Red Flags

| Red Flag | Problem | Solution |
|----------|---------|----------|
| R² = 0.95, but predictions = mean(y) ± noise | Overfitting to noise | Check cross-validated Q²; reduce features |
| RMSE < measurement noise | Perfect fit (suspicious) | Check for data leakage or duplicate samples |
| Predictions outside physical range | Model extrapolating | Add domain constraints (e.g., clip to [0, 100]%) |
| Residuals strongly patterned (not random) | Model misspecification | Add nonlinear terms, interaction terms |

### Statistical Red Flags

| Red Flag | Problem | Solution |
|----------|---------|----------|
| p = 0.049 (just below 0.05) | p-hacking or cherry-picking | Report effect size + CI; pre-register analysis |
| Significant p-value but Cohen's d < 0.2 | Large sample, trivial effect | Focus on effect size, not p-value |
| Non-significant but Cohen's d > 0.8 | Underpowered study | Increase sample size; report CI on effect size |
| 50 tests, 3 "significant" at α=0.05 | False positives by chance | Apply Bonferroni or FDR correction |

---

## Domain-Specific Thresholds

### Food Spectroscopy (FTIR/Raman/NIR)

| Application | Metric | Acceptable | Good | Excellent |
|-------------|--------|------------|------|-----------|
| **Oil Authentication** | AUC | >0.85 | >0.90 | >0.95 |
| **Adulteration Detection** | Recall | >0.90 | >0.95 | >0.98 |
| **Quantitative Analysis (e.g., fat%)** | RMSE | <1% | <0.5% | <0.2% |
| **Heating Quality** | R² | >0.70 | >0.80 | >0.90 |
| **Batch QC** | Specificity | >0.95 | >0.98 | >0.99 |

### Compliance & Regulatory Standards

| Regulation | Requirement | FoodSpec Metric | Threshold |
|------------|-------------|-----------------|-----------|
| **FDA 21 CFR Part 11** | Validated methods | Cross-validated AUC | >0.90 |
| **ISO 17025** | Measurement uncertainty | RMSE / σ_reference | <0.20 |
| **AOAC Guidelines** | Recovery | Recall (sensitivity) | >0.95 |
| **EU Regulation 2017/625** | False positive rate | 1 - Specificity | <0.05 |

---

## Metric Selection Guide

### Choose Metrics Based on Task

| Task | Primary Metric | Secondary Metrics | Avoid |
|------|----------------|-------------------|-------|
| **Binary Classification (balanced)** | AUC-ROC, F1 | Accuracy, MCC | Precision or recall alone |
| **Binary Classification (imbalanced)** | AUC-PR, Balanced Accuracy | F1, MCC, Cohen's Kappa | Accuracy |
| **Multiclass Classification** | Macro F1, Cohen's Kappa | Per-class F1, confusion matrix | Micro F1 (same as accuracy) |
| **Quantitative Regression** | R², RMSE | MAE, Q² (CV) | MAPE (if y=0 possible) |
| **Mixture Analysis** | RMSE, MAPE | R², Bland-Altman | Single-point accuracy |
| **Hypothesis Testing** | p-value + Effect size | Confidence intervals | p-value alone |

---

## Reporting Checklist

When reporting metrics, **ALWAYS** include:

- [x] **Primary metric with 95% CI** (e.g., AUC = 0.92 [0.87, 0.96])
- [x] **Sample size** (n_train, n_test, n_folds if CV)
- [x] **Class balance** (for classification: ratio, counts per class)
- [x] **Validation strategy** (e.g., stratified 5-fold CV, holdout 20%)
- [x] **Effect size** (for hypothesis tests: Cohen's d, η², etc.)
- [x] **Context interpretation** ("Excellent for balanced data" or "Fair given small n")
- [x] **Comparison to baseline** (e.g., "20% improvement over random classifier")

---

## References

1. **Cohen, J. (1988).** Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Routledge.
2. **Eilers, P. H., & Boelens, H. F. (2005).** Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre Report.
3. **Japkowicz, N., & Shah, M. (2011).** Evaluating Learning Algorithms: A Classification Perspective. Cambridge University Press.
4. **Wasserstein, R. L., & Lazar, N. A. (2016).** The ASA Statement on p-Values: Context, Process, and Purpose. The American Statistician, 70(2), 129-133.
5. **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. Journal of the Royal Statistical Society: Series B, 57(1), 289-300.

---

## See Also

- **[Metrics & Evaluation](../reference/metrics_reference.md)** — Overview of all metrics
- **[Model Evaluation](../methods/chemometrics/model_evaluation_and_validation.md)** — Cross-validation strategies
- **[Hypothesis Testing](../methods/statistics/hypothesis_testing_in_food_spectroscopy.md)** — Statistical tests and p-values
- **[Statistical Power](../protocols/statistical_power_and_limits.md)** — Sample size planning
- **[T-tests & Effect Sizes](../methods/statistics/t_tests_effect_sizes_and_power.md)** — Detailed effect size guide
