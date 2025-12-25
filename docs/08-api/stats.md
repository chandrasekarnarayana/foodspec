# Stats API Reference

Statistical analysis module for hypothesis testing, effect sizes, and correlations.

## run_ttest

Independent or paired t-tests with effect sizes.

::: foodspec.stats.hypothesis_tests.run_ttest
    options:
      show_source: false
      heading_level: 3

## run_anova

One-way ANOVA with effect sizes.

::: foodspec.stats.hypothesis_tests.run_anova
    options:
      show_source: false
      heading_level: 3

## benjamini_hochberg

Benjamini-Hochberg FDR correction for multiple testing.

::: foodspec.stats.hypothesis_tests.benjamini_hochberg
    options:
      show_source: false
      heading_level: 3

## compute_cohens_d

Cohen's d effect size for two-group comparisons.

::: foodspec.stats.effects.compute_cohens_d
    options:
      show_source: false
      heading_level: 3

## compute_correlations

Correlation analysis between variables.

::: foodspec.stats.correlations.compute_correlations
    options:
      show_source: false
      heading_level: 3

---

## Usage Examples

### T-Test with Effect Size

```python
from foodspec.stats import run_ttest, compute_cohens_d

# Compare two groups
result = run_ttest(group_A, group_B)
effect = compute_cohens_d(group_A, group_B)

print(f"t = {result.statistic:.3f}, p = {result.pvalue:.4f}")
print(f"Cohen's d = {effect:.3f}")
```

### Multiple Testing Correction

```python
from foodspec.stats import benjamini_hochberg

# Correct p-values from multiple tests
p_values = [0.01, 0.03, 0.05, 0.12, 0.45]
corrected = benjamini_hochberg(p_values, alpha=0.05)

print(corrected)
```
