# Stats API Reference

!!! info "Module Purpose"
    Statistical hypothesis testing, effect sizes, and correlation analysis for spectral data.

---

## Quick Navigation

| Function | Purpose | Use Case |
|----------|---------|----------|
| [`run_ttest()`](#hypothesis-testing) | t-test with effect sizes | Compare two groups |
| [`run_anova()`](#hypothesis-testing) | ANOVA with effect sizes | Compare multiple groups |
| [`benjamini_hochberg()`](#multiple-testing) | FDR correction | Control false discoveries |
| [`compute_cohens_d()`](#effect-sizes) | Cohen's d | Measure effect magnitude |
| [`compute_correlations()`](#correlations) | Correlation analysis | Feature relationships |

---

## Common Patterns

### Pattern 1: Group Comparison with Effect Sizes

```python
from foodspec.stats import run_ttest, compute_cohens_d

# Compare two varieties
group_A = fs[fs.metadata['variety'] == 'A'].x[:, peak_idx]
group_B = fs[fs.metadata['variety'] == 'B'].x[:, peak_idx]

# t-test
result = run_ttest(group_A, group_B)
print(f"t-statistic: {result.statistic:.3f}")
print(f"p-value: {result.pvalue:.4f}")

# Effect size
d = compute_cohens_d(group_A, group_B)
print(f"Cohen's d: {d:.3f}")
```

### Pattern 2: Multiple Group Comparison

```python
from foodspec.stats import run_anova, benjamini_hochberg

# ANOVA across varieties
groups = [fs[fs.metadata['variety'] == v].x[:, peak_idx] 
          for v in ['A', 'B', 'C']]

result = run_anova(*groups)
print(f"F-statistic: {result.statistic:.3f}")
print(f"p-value: {result.pvalue:.4f}")

# Multiple testing correction
p_values = [0.001, 0.045, 0.12, 0.002]
corrected = benjamini_hochberg(p_values, alpha=0.05)
print(f"Significant after correction: {corrected.sum()}")
```

### Pattern 3: Feature Correlation Analysis

```python
from foodspec.stats import compute_correlations

# Correlate spectral features with quality score
correlations = compute_correlations(
    fs.x,
    fs.metadata['quality_score'],
    method='pearson'
)

# Find most correlated wavenumbers
top_features = correlations.argsort()[-10:]
print(f"Top correlated wavenumbers: {fs.wavenumbers[top_features]}")
```

---

## Hypothesis Testing

### run_ttest

Independent or paired t-tests with effect sizes.

::: foodspec.stats.hypothesis_tests.run_ttest
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.stats import run_ttest

# Independent t-test
result = run_ttest(group1, group2, paired=False)
if result.pvalue < 0.05:
    print("Significantly different")
```

### run_anova

One-way ANOVA with effect sizes.

::: foodspec.stats.hypothesis_tests.run_anova
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.stats import run_anova

# ANOVA
result = run_anova(group1, group2, group3)
print(f"F={result.statistic:.2f}, p={result.pvalue:.4f}")
```

---

## Multiple Testing

### benjamini_hochberg

Benjamini-Hochberg FDR correction for multiple testing.

::: foodspec.stats.hypothesis_tests.benjamini_hochberg
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.stats import benjamini_hochberg

# Correct multiple p-values
p_values = [0.001, 0.03, 0.08, 0.15, 0.002]
significant = benjamini_hochberg(p_values, alpha=0.05)
print(f"Significant: {significant}")
```

---

## Effect Sizes

### compute_cohens_d

Cohen's d effect size for two-group comparisons.

::: foodspec.stats.effects.compute_cohens_d
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.stats import compute_cohens_d

# Effect size
d = compute_cohens_d(treatment, control)

# Interpretation:
# |d| < 0.2: Small effect
# |d| < 0.5: Medium effect
# |d| >= 0.8: Large effect
print(f"Effect size: {d:.3f} ({'small' if abs(d)<0.5 else 'large'})")
```

---

## Correlations

### compute_correlations

Correlation analysis between variables.

::: foodspec.stats.correlations.compute_correlations
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.stats import compute_correlations

# Pearson correlation
r = compute_correlations(X, y, method='pearson')

# Spearman (non-parametric)
rho = compute_correlations(X, y, method='spearman')
```

---

## Cross-References

**Related Modules:**
- [Metrics](metrics.md) - Model evaluation metrics
- [Chemometrics](chemometrics.md) - Statistical modeling

**Related Workflows:**
- [Statistical Validation](../workflows/standard_templates.md) - Hypothesis testing examples

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
