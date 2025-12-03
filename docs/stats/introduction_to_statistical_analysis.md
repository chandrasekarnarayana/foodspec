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

## Further reading
- [Hypothesis testing](hypothesis_testing_in_food_spectroscopy.md)
- [ANOVA and MANOVA](anova_and_manova.md)
- [t-tests, effect sizes, power](t_tests_effect_sizes_and_power.md)
- [Study design](study_design_and_data_requirements.md)
- API: [Statistics](../api/stats.md)
