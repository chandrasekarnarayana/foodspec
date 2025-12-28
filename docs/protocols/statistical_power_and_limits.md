# Statistical Power and Study Design Limits

**Who:**
Study planners, researchers designing FoodSpec experiments, statisticians and reviewers evaluating study robustness.

**What:**
Practical guidance on sample size, replicates, batch effects, and statistical power specific to food spectroscopy applications.

**When:**
Use before starting a new FoodSpec study to determine adequate sample size and replication strategy.

**When NOT:**
This is not a substitute for formal statistical power analysis (consult a statistician for complex designs); nor is it a replacement for domain-specific method validation.

**Key Assumptions:**
- Studies follow the [Reference Protocol](reference_protocol.md) structure
- Samples are representative of intended population
- You have reasonable estimates of effect size (from prior studies or pilot data)

**What can go wrong:**
- Underpowered studies → false negatives, unreliable estimates
- Overconfident sample size estimates → overfitting, poor generalization
- Unmanaged batch effects → spurious results
- Ignoring multiple testing → inflated Type I error

---

## Quick Rules of Thumb for Food Spectroscopy Studies

### Minimum Sample Sizes

| Study Type | Minimum n per Class | Rationale | Notes |
|-----------|-------------------|-----------|-------|
| **Proof-of-concept / feasibility** | 15 | Small pilot; hypothesis generation only | High risk of overfitting; don't use for claims |
| **Classification (exploratory)** | 30 | Standard ML minimum; allows nested CV | Recommended starting point |
| **Classification (confirmatory)** | 50–100 | Supports rigorous cross-validation | Larger test set; robust metrics |
| **Regression (exploratory)** | 30–50 | Depends on feature count; rule: n ≥ 5 × n_features | Increase if high-dimensional |
| **Regression (confirmatory)** | 100–200 | Higher power; validates on diverse test set | Gold standard |

### Replication Strategy

**Standard recommendation: 3 replicates per sample, 3 levels of variation:**

```yaml
For each biological sample:
  
  Level 1: Technical replicates (same vial, immediate rescans)
    → n_technical = 3
    → Captures instrument noise
  
  Level 2: Intra-day replicates (same sample, re-mounted)
    → n_intra_day = 3
    → Captures sample/mounting variability
  
  Level 3: Inter-day replicates (same sample, different days)
    → n_inter_day = 2–3
    → Captures temporal drift
  
  Total spectra per sample = 3 × 3 × 2 = 18 (or 3 × 3 × 3 = 27)
  
For n = 30 samples: total spectra = 30 × 18 = 540 (manageable)
For n = 100 samples: total spectra = 100 × 18 = 1800 (feasible; time-intensive)
```

**Minimal replication (if resources constrained):**

```yaml
For each sample: 3 technical replicates only
  → Captures instrument noise
  → Does NOT capture biological/mounting variability
  → Use only for very robust discrimination (e.g., oils vs. non-oils)
  → Risk: Model may overfit to instrument-specific features
```

---

## Sample Size Calculation: Formula

### For Classification (Diagnostic Accuracy)

If targeting a specific **sensitivity** (true positive rate) or **specificity** (true negative rate):

```plaintext
n = (z_α/2 + z_β)² × p(1-p) / d²

where:
  z_α/2 = critical value for significance level α (α=0.05 → z=1.96)
  z_β = critical value for power (1-β); power=0.80 → z=0.84; power=0.90 → z=1.28
  p = expected sensitivity/specificity (0–1)
  d = acceptable margin of error (e.g., d=0.10 for ±10%)

Example: Achieve 90% sensitivity ± 10% with 80% power
  n = (1.96 + 0.84)² × 0.90 × 0.10 / 0.10²
  n = 7.84 × 0.09 / 0.01
  n ≈ 71 samples (per class)
```

### For Regression (Predicting Continuous Outcome)

If targeting a specific **R² or correlation coefficient:**

```yaml
n = (z_α/2 + z_β)² / (2 × r²)

where:
  r = expected correlation coefficient (0–1)
  other terms as above

Example: Detect correlation r=0.5 with 80% power
  n = (1.96 + 0.84)² / (2 × 0.5²)
  n = 7.84 / 0.5
  n ≈ 16 samples (conservative; increase to ≥30 for robustness)
```

**Practical note:** Add 10–20% buffer for dropouts/quality control failures.

---

## Batch Effects and Confounding

### What Are Batch Effects?

Systematic variations in spectral data caused by:

| Source | Impact | Example |
|--------|--------|---------|
| **Instrument calibration** | Baseline shifts, intensity scaling | Raman laser power drifts 5% over day |
| **Temperature** | Peak shifts (~0.1–0.5 cm⁻¹/°C) | Room temp varies 20–24 °C |
| **Operator** | Sampling technique, vial mounting | Different mounting pressure affects baseline |
| **Sample storage** | Oxidation, phase changes, settling | Oil exposed to light for 1 month before analysis |
| **Time** | Detector aging, baseline drift | Spectra acquired 6 months apart |
| **Instrument geometry** | Laser focus, fiber coupling | Change focus position by 0.5 mm |

**Magnitude:** Batch effects can exceed biological signal; e.g., a 2% intensity shift from temperature variation may obscure a 1% difference due to sample composition.

### Managing Batch Effects

#### Strategy 1: Design Orthogonality (Preferred)

**Randomize batch assignment:**
- Assign samples to acquisition batches (days/instruments) randomly
- Avoid confounding: don't put all "authentic" samples on day 1 and "adulterated" on day 2
- Include **batch controls** (same reference material) in every batch

```yaml
Example randomization:
  Batch 1 (Day 1): 5 authentic, 5 adulterated, 1 reference material
  Batch 2 (Day 2): 5 authentic, 5 adulterated, 1 reference material
  Batch 3 (Day 1, different operator): 5 authentic, 5 adulterated, 1 reference material
  ...
```

#### Strategy 2: Batch-Aware Cross-Validation

Use stratified CV that respects batch structure:

```python
from sklearn.model_selection import GroupKFold

# Each batch is a group
group_cv = GroupKFold(n_splits=5)

for train_idx, test_idx in group_cv.split(X, y, groups=batch_ids):
    # No data from the same batch in train and test
    # Better generalization to new batches
```

#### Strategy 3: Batch Correction (Advanced)

If batch effects unmanaged, use statistical batch correction:

```yaml
ComBat or SVA (Surrogate Variable Analysis):
  1. Fit a linear model: spectrum ~ class + batch
  2. Estimate batch coefficients
  3. Remove batch contribution: spectrum_corrected = spectrum - batch_effect
  
Caveats:
  - Assumes batch effect is additive (true for most cases)
  - Requires ≥10 samples per batch for reliable estimation
  - May remove true biological variation if confounded with batch
```

---

## Multiple Testing and Multiple Comparisons

### The Problem

If testing **k independent hypotheses** with significance level α per test, the **family-wise error rate (FWER)** is:

```yaml
FWER ≥ 1 - (1 - α)^k

Example: 10 independent tests, α=0.05 per test
  FWER ≥ 1 - (1 - 0.05)^10 = 1 - 0.59 = 0.41
  → 41% chance of at least one false positive!
```

### Solutions

| Method | When to Use | Conservativeness |
|--------|-----------|-----------------|
| **Bonferroni** | Few tests (k < 10) | Very conservative; high false negative rate |
| **Holm–Bonferroni** | Few tests (k < 10) | Less conservative than Bonferroni |
| **FDR (Benjamini–Hochberg)** | Many tests (k > 10); exploratory | Balanced; controls expected proportion of false positives |
| **Permutation test** | No distributional assumptions | Flexible; computationally intensive |

**Recommendation for FoodSpec:**

- If comparing k spectral bands or features for significance: use **FDR correction** (q < 0.05)
- If performing k cross-validation folds: no correction needed (folds not independent tests)
- If fitting k models and reporting best: use **nested CV** to avoid selection bias (no correction needed)

---

## Effect Size and Clinical Significance

### Distinguishing Statistical vs. Practical Significance

A large sample can detect tiny differences:

```yaml
Example: Comparing oil authenticity scores

Method A (n=10 per group):
  Authentic: mean=90, SD=5
  Adulterated: mean=88, SD=5
  t-test: p=0.13 (not significant)
  
Method B (n=1000 per group):
  Authentic: mean=90, SD=5
  Adulterated: mean=88, SD=5
  t-test: p=0.001 (significant!)
  
But the 2-point difference is tiny—does it matter?
  → Effect size (Cohen's d) = 2/5 = 0.4 (small–medium)
  → Decision: Statistically significant but may not be practically meaningful
```

**Always report effect sizes:**

| Measure | Interpretation |
|---------|----------------|
| Cohen's d | 0.2 = small; 0.5 = medium; 0.8 = large |
| Cohen's f (ANOVA) | 0.1 = small; 0.25 = medium; 0.4 = large |
| Cramér's V (categorical) | 0.1 = small; 0.3 = medium; 0.5 = large |
| AUC (diagnostic) | 0.7–0.8 = fair; 0.8–0.9 = good; >0.9 = excellent |

---

## Study Design Checklist

- [ ] **Sample size justified** (power analysis or literature-based)
- [ ] **Replication strategy defined** (technical, intra-day, inter-day; minimum 3 replicates)
- [ ] **Batch randomization** (samples assigned randomly to batches, not confounded by class)
- [ ] **Batch controls** (reference material in every batch)
- [ ] **Temporal separation** (training and test from different dates/batches if possible)
- [ ] **Operator blinding** (if feasible; prevents conscious/unconscious bias)
- [ ] **Exclusion criteria pre-defined** (what spectra are unacceptable? why?)
- [ ] **Validation strategy selected** (nested CV, hold-out test set, external cohort, or combination)
- [ ] **Multiple testing correction** (if k > 5 independent tests, apply FDR or permutation test)
- [ ] **Effect sizes calculated** (not just p-values)
- [ ] **Metadata logged** (operator, date, temperature, instrument settings, storage conditions)

---

## When Statistical Assumptions Are Violated

### Non-Normality

**Spectral intensity distributions are often non-normal** (skewed, multimodal).

| Consequence | Solution |
|------------|----------|
| t-tests unreliable; confidence intervals wrong | Use non-parametric tests (Mann–Whitney U, Kruskal–Wallis) or permutation tests |
| Linear regression biased | Use robust regression (Huber, RANSAC) or quantile regression |
| PCA assumes normality | Use ICA or other non-normal component models (usually not necessary in practice) |

### Heteroscedasticity (Unequal Variance)

**Spectral data often have intensity-dependent noise** (SNR improves with peak height).

| Consequence | Solution |
|------------|----------|
| t-tests and ANOVA unreliable | Use Welch's t-test (doesn't assume equal variance) or robust methods |
| Confidence intervals wrong | Use bootstrap or permutation CIs |
| Linear regression biased | Use weighted least squares or robust regression |

### Autocorrelation (Within-Sample Dependence)

**Replicates from the same sample are correlated.**

| Consequence | Solution |
|------------|----------|
| Sample size overestimated; false confidence | Use mixed-effects models (sample as random effect) |
| Standard errors understated | Compute clustered SEs (by sample) |
| Simple CV unreliable | Use grouped/leave-one-group-out CV (by sample) |

---

## Recommended Reading

- [Study Design and Data Requirements](../methods/statistics/study_design_and_data_requirements.md) — Deeper theory
- [Hypothesis Testing](../methods/statistics/hypothesis_testing_in_food_spectroscopy.md) — Testing principles
- [t-Tests, Effect Sizes, and Power](../methods/statistics/t_tests_effect_sizes_and_power.md) — Detailed formulas
- [Reference Protocol](reference_protocol.md) — Study structure

---

## Quick Reference: Sample Size Table (Classification)

Minimum n per class for 80% power, α=0.05, assuming expected sensitivity/specificity of 0.85:

| Margin of Error | Required n |
|-----------------|-----------|
| ±20% (rough estimate) | 9 |
| ±15% (typical) | 15 |
| ±10% (careful study) | 33 |
| ±7% (rigorous study) | 68 |
| ±5% (gold standard) | 138 |

**Rule of thumb:** Aim for ≥30–50 per class as default; increase to ≥100 if claims are high-stakes or if effect size uncertain.

