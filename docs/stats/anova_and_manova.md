# ANOVA and MANOVA for Food Spectroscopy

ANOVA tests whether group means differ; MANOVA extends this to multiple dependent variables (e.g., multiple peak ratios or PC scores). This chapter details both, with food spectroscopy examples.

## ANOVA basics
- **One-way ANOVA:** Compares means across ≥3 groups.
  - F-statistic: \( F = \frac{SS_\text{between}/(k-1)}{SS_\text{within}/(N-k)} \)
    - \( SS_\text{between} \): sum of squares between groups; \( SS_\text{within} \): within groups; \( k \): groups; \( N \): total samples.
  - Assumptions: normality of residuals, homoscedasticity, independence.
- **Two-way ANOVA (outline):** Two factors (e.g., oil_type and batch) and interaction; requires balanced or at least sufficiently populated cells.

## MANOVA
- Tests multivariate mean differences across groups (e.g., multiple ratios/PCs).
  - Uses Wilks’ Lambda, Pillai’s trace, etc.
  - Assumptions: multivariate normality, homogeneity of covariance matrices, independence.
  - Implemented via statsmodels (optional dependency).

## Post-hoc comparisons
- **Tukey HSD:** Pairwise group comparisons controlling family-wise error.
- **Other corrections:** Bonferroni/FDR (not yet implemented; can be done externally if needed).

## Effect sizes
- Eta-squared \( \eta^2 = SS_\text{between} / SS_\text{total} \).
- Partial eta-squared \( = SS_\text{between} / (SS_\text{between} + SS_\text{within}) \).
- Report alongside p-values to convey magnitude.

## Food spectroscopy examples
- Oil authentication: ANOVA on 1655/1742 ratio across oil types; Tukey to see which pairs differ.
- Heating stages: ANOVA on unsaturation ratio across time bins; partial eta-squared to quantify effect size.
- MANOVA: Multiple ratios or PC scores across microbial strains.

![Boxplot with clear group differences (ANOVA illustration)](../assets/boxplot_anova.png)

## Code snippets
```python
from foodspec.stats import run_anova, run_tukey_hsd, compute_anova_effect_sizes

# df with columns ratio, oil_type
res = run_anova(df["ratio"], df["oil_type"])
print(res.summary)

# Effect size (requires sums of squares; here approximated)
ss_between = 10.0
ss_total = 20.0
print(compute_anova_effect_sizes(ss_between, ss_total))

# Tukey (if statsmodels installed)
try:
    tukey = run_tukey_hsd(df["ratio"], df["oil_type"])
    print(tukey.head())
except ImportError:
    pass
```

## Decision aid: Is ANOVA appropriate?
```mermaid
flowchart TD
  A[Groups >= 3?] --> B{Normality/homoscedasticity reasonable?}
  B -->|Yes| C[ANOVA → Tukey]
  B -->|No| D{Transform or nonparametric?}
  D -->|Transform| E[Reassess assumptions]
  D -->|Nonparametric| F[Kruskal–Wallis (TODO)]
```

## Reporting (MethodsX tone)
- State test type (ANOVA/MANOVA), factors, assumptions, effect sizes.
- Provide p-values and post-hoc results; include means/SD per group.
- Visuals: boxplots/mean plots with CIs; Tukey pairwise table/plot.

## Further reading
- [Hypothesis testing](hypothesis_testing_in_food_spectroscopy.md)
- [t-tests, effect sizes, power](t_tests_effect_sizes_and_power.md)
- [Study design](study_design_and_data_requirements.md)
- API: [Statistics](../api/stats.md)
