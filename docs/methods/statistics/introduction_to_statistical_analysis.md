# Introduction to Statistical Analysis in Food Spectroscopy

Statistical analysis complements chemometrics and machine learning by testing hypotheses, quantifying uncertainty, and framing results in a way reviewers and regulators expect. This chapter situates classical statistics within Raman/FTIR/NIR workflows in FoodSpec.

## Why statistics matters
- **Validation:** Confirm that observed differences (e.g., between oil types) are unlikely to be random.
- **Interpretation:** Link spectral features (peaks/ratios/PCs) to food science questions (authenticity, degradation).
- **Reporting:** Provide p-values, confidence intervals, and effect sizes alongside ML metrics for rigor and reproducibility.

## Data types in FoodSpec
- **Raw spectra:** Intensity vs wavenumber (cm⁻¹).
- **Derived features:** Peak heights/areas, band integrals, ratios, PCA scores, mixture coefficients.
- **Metadata:** Group labels (oil_type), time/temperature (heating), batches/instruments.

## Where tests fit in workflows
- **Oil authentication:** ANOVA/Tukey on ratios or PC scores across oil types; t-tests on binary comparisons.
- **Heating quality:** Correlation of ratios vs time; ANOVA across stages.
- **Mixture analysis:** MANOVA/ANOVA on mixture proportions vs spectral features.
- **Batch QC:** Tests comparing reference vs suspect sets; correlation maps.

## Assumptions and preprocessing
- Many tests assume approximate normality, homoscedasticity, and independence.
- Good preprocessing (baseline, normalization, scatter correction) reduces artifacts that violate assumptions.
- When assumptions fail, consider nonparametric tests or robust designs (see [Nonparametric methods](nonparametric_methods_and_robustness.md)).

## Quick example
```python
import pandas as pd
from foodspec.stats import run_anova

df = pd.DataFrame({"ratio": [1.0, 1.1, 0.9, 1.8, 1.7, 1.9],
                   "oil_type": ["olive", "olive", "olive", "sunflower", "sunflower", "sunflower"]})
res = run_anova(df["ratio"], df["oil_type"])
print(res.summary)
```

## Decision aid: tests vs models
```mermaid
flowchart LR
  A[Question] --> B{Compare means?}
  B -->|Yes| C{Groups > 2?}
  C -->|No| D[t-test]
  C -->|Yes| E[ANOVA/MANOVA + post-hoc]
  B -->|No| F{Association?}
  F -->|Yes| G[Correlation (Pearson/Spearman)]
  F -->|No| H[Predictive modeling (see ML chapters)]
```

---

## When Results Cannot Be Trusted

⚠️ **Red flags for statistical analysis validity across all methods:**

1. **Assumptions not checked before test selection**
   - Each test (t-test, ANOVA, correlation) assumes normality, independence, or linearity
   - Violating assumptions without correcting inflates Type I error
   - **Fix:** Always check Q-Q plots, Shapiro–Wilk, Levene's test; use robust/nonparametric alternatives

2. **P-value interpreted as truth (p = 0.04 → "result is true with 96% confidence")**
   - p-value is probability of observing data IF null hypothesis is true; not probability that result is true
   - Misinterpretation is widespread and leads to overconfidence
   - **Fix:** Report effect size and confidence intervals; avoid binary "significant/not significant" language

3. **Multiple testing without correction (testing 50 hypotheses at α = 0.05, expecting ≤2.5 false positives by chance)**
   - Uncorrected p-values are misleading when many tests performed
   - Problem scales with number of tests
   - **Fix:** Declare hypotheses a priori; apply Bonferroni, FDR, or permutation-based correction

4. **Sample size chosen arbitrarily ("n = 10 seems reasonable") without power analysis**
   - Underpowered studies miss real effects and inflate false negatives
   - No rationale for sample choice reduces credibility
   - **Fix:** Conduct a priori power analysis based on target effect size and power (0.80)

5. **Preprocessing choices undisclosed (baseline correction, outlier removal, transformation)**
   - Different preprocessing → different results, even on same raw data
   - Undisclosed choices enable hidden p-hacking
   - **Fix:** Freeze preprocessing before analysis; document all choices; report sensitivity to preprocessing

6. **Batch confounding (treatment A = Day 1 analyzer, treatment B = Day 2 analyzer)**
   - Systematic batch effects mimick biological differences
   - Impossible to know whether test detects biology or batch artifact
   - **Fix:** Randomize sample order; include batch in model; use batch-aware CV

7. **Selective reporting of results (reporting 3 significant tests out of 20 performed)**
   - Bias toward positive results inflates false positive rate
   - Non-significant results are equally informative
   - **Fix:** Pre-register analyses; report all tests performed; use exploratory vs confirmatory designations

8. **Independence assumption violated (using technical replicates as if independent biological replicates)**
   - Repeated measurements of same unit are autocorrelated
   - Tests assuming independence produce inflated significance
   - **Fix:** Analyze ≥3 distinct samples; document which measurements are replicates; use mixed-effects models if nested

## Further reading
- [Hypothesis testing](hypothesis_testing_in_food_spectroscopy.md)
- [ANOVA and MANOVA](anova_and_manova.md)
- [t-tests, effect sizes, power](t_tests_effect_sizes_and_power.md)
- [Study design](study_design_and_data_requirements.md)
- API: [Statistics](../../api/stats.md)

---

## Next Steps

- [Hypothesis testing](hypothesis_testing_in_food_spectroscopy.md) — Statistical significance framework.
- [Study design](study_design_and_data_requirements.md) — Plan robust experiments.
- [API: Statistics](../../api/stats.md) — Full reference for all statistical functions.
