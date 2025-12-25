# Hypothesis Testing in Food Spectroscopy

Hypothesis tests answer questions like “Are mean ratios different across oil types?” or “Did heating change this band?” This chapter summarizes common tests, assumptions, and interpretation for spectral features and derived metrics.

## Core tests and questions
- **One-sample t-test:** Is a mean different from a hypothesized value (e.g., expected ratio = 1.0)?
  - Statistic: \( t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \)
- **Two-sample t-test (independent):** Are two group means different (e.g., olive vs sunflower ratios)?
  - Welch’s variant (unequal variances) is default in FoodSpec.
- **Paired t-test:** Are paired measurements different (e.g., before/after heating the same sample)?
  - Apply to paired observations to control subject-level variability.
- **One-way ANOVA:** Are there differences among ≥3 groups (oil types, heating stages)?
  - F-statistic from between-group vs within-group variance.
- **MANOVA:** Are multivariate means (e.g., multiple ratios/PCs) different across groups?
  - Uses Wilks’ Lambda, Pillai’s trace, etc. (statsmodels required).
- **Post-hoc (Tukey HSD):** Which pairs differ after ANOVA?
- **Nonparametric:** Kruskal–Wallis (≥3 groups) or Mann–Whitney/Wilcoxon (2 groups) when normality/variance assumptions are violated.

## Assumptions (and when not to use)
| Assumption | Applies to | Diagnose | If violated |
| --- | --- | --- | --- |
| Normality of residuals | t-tests/ANOVA | QQ-plots, Shapiro on residuals | Transform, or use nonparametric |
| Homoscedasticity | ANOVA, pooled t-test | Levene/Bartlett, residual vs fit plots | Welch t-test, transform, nonparametric |
| Independence | All | Study design, randomized acquisition | Re-acquire or block; avoid paired tests on independent data |
| Sufficient group size | All | Group counts | Use caution with very small n; report effect sizes/CI |

## Interpretation
- **p-value:** Probability of observing the statistic (or more extreme) under the null hypothesis. Small p suggests group differences.
- **Effect size:** Magnitude matters; report Cohen’s d or eta-squared/partial eta-squared (see [t-tests, effect sizes, power](t_tests_effect_sizes_and_power.md)).
- **Practical relevance:** A statistically significant difference may be small; tie back to sensory/quality thresholds.

## Food spectroscopy examples
- Compare 1655/1742 ratio across oil types (ANOVA + Tukey).
- Paired t-test on spectra before vs after heating the same oil.
- MANOVA on multiple peak ratios/PC scores across species (microbial or meat ID).

## Decision aid: Which test?
```mermaid
flowchart LR
  A[Compare means?] --> B{Groups = 1?}
  B -->|Yes| C[One-sample t-test]
  B -->|No| D{Groups = 2?}
  D -->|Yes| E{Paired?}
  E -->|Yes| F[Paired t-test]
  E -->|No| G[Two-sample t-test (Welch)]
  D -->|No| H{Multivariate?}
  H -->|Yes| I[MANOVA (statsmodels)]
  H -->|No| J[One-way ANOVA → Tukey HSD]
```

## Code snippets (FoodSpec)
```python
import pandas as pd
from foodspec.stats import run_anova, run_ttest, run_tukey_hsd

df = pd.DataFrame({
    "ratio": [1.0, 1.1, 0.9, 1.8, 1.7, 1.9],
    "oil_type": ["olive", "olive", "olive", "sunflower", "sunflower", "sunflower"]
})
anova_res = run_anova(df["ratio"], df["oil_type"])
print(anova_res.summary)

tt_res = run_ttest(df[df["oil_type"]=="olive"]["ratio"],
                   df[df["oil_type"]=="sunflower"]["ratio"])
print(tt_res.summary)

# Tukey (if statsmodels installed)
try:
    tukey_tbl = run_tukey_hsd(df["ratio"], df["oil_type"])
    print(tukey_tbl.head())
except ImportError:
    pass
```

## Summary
- Choose t-tests for two groups (paired/unpaired), ANOVA for ≥3 groups, MANOVA for multivariate responses.
- Check assumptions; consider nonparametric tests if violated.
- Report p-values and effect sizes; discuss practical relevance for food quality/authenticity.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for hypothesis testing validity:**

1. **P-value without context (reporting p = 0.04 without effect size, n, or assumptions)**
   - p-value measures evidence against H₀, not size or relevance of effect
   - Small p + tiny effect size = statistically significant but practically irrelevant
   - **Fix:** Always report effect size (Cohen's d, R²), sample size, and confidence intervals

2. **Assumptions not checked (running t-test on non-normal data)**
   - Parametric tests assume normality, independence, homoscedasticity
   - Violating assumptions inflates Type I error (false positives)
   - **Fix:** Check Q-Q plots, Shapiro–Wilk, Levene's test; use nonparametric tests if violated

3. **Undisclosed multiple testing (tested 100 peaks at p < 0.05, reported 3 significant without correction)**
   - Each test has 5% false positive rate; 100 tests expect 5 false positives by chance
   - Uncorrected p-values are misleading when many tests performed
   - **Fix:** Declare tests a priori; apply Bonferroni or FDR correction; report adjusted p-values

4. **Ignoring batch effects (all diseased samples = Day 1 analyzer, all healthy = Day 2)**
   - Group differences may be confounded with systematic batch drift
   - Impossible to know whether test detects biology or batch artifact
   - **Fix:** Randomize sample order; include batch in model; use batch-aware CV

5. **Sample size chosen post-hoc ("n=5 seems reasonable") without power analysis**
   - Arbitrary sample sizes often too small for target power (0.80)
   - Underpowered tests miss real effects and inflate false negatives
   - **Fix:** Conduct a priori power analysis; determine n before collecting data

6. **Selective reporting (reporting only significant contrasts out of many)**
   - Post-hoc hunting for significance inflates Type I error
   - p-value interpretation assumes pre-specified hypothesis, not post-hoc exploration
   - **Fix:** Declare hypotheses before analysis; use exploratory data analysis separately from confirmatory tests

7. **Interpreting non-significance as equivalence (p > 0.05 means "no difference")**
   - Failure to reject H₀ ≠ proof of equality
   - May reflect low power, not absence of effect
   - **Fix:** Report confidence intervals; compute post-hoc power; consider equivalence tests (e.g., TOST)

8. **Pseudo-replication (20 technical replicates of 1 sample treated as 20 independent observations)**
   - Repeated measurements of same unit are autocorrelated
   - Tests assuming independence produce inflated false positives
   - **Fix:** Analyze ≥3 distinct samples; report which measurements are technical replicates

## Further reading
- [ANOVA and MANOVA](anova_and_manova.md)
- [t-tests, effect sizes, power](t_tests_effect_sizes_and_power.md)
- [Correlation and mapping](correlation_and_mapping.md)
- API: [Statistics](../api/stats.md)
