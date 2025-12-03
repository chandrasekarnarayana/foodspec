# Statistical tests for food spectroscopy

You are here: Methods & theory → Statistical tests for food spectroscopy

Questions this page answers
- Why do statistical tests matter in Raman/FTIR food studies?
- Which tests answer which questions, and what are their assumptions?
- How do I run common tests with SciPy on foodspec-derived features?
- How should I report test results in a paper?

> Status: Legacy/archived. The maintained statistics content lives under `../stats/`. FoodSpec provides spectral features; statistical hypothesis tests are run with SciPy/statsmodels. Dependencies: SciPy (https://scipy.org/), statsmodels (https://www.statsmodels.org/)

## Introduction
Statistical tests help determine whether observed spectral differences or trends are unlikely to be due to chance. In Raman/FTIR food studies, they support claims about authenticity, treatment effects (heating, storage), or mixture proportions.

## Basic concepts
- **Null hypothesis (H0)**: no difference/effect/association.
- **p-value**: probability of observing data as extreme as yours if H0 is true.
- **Effect size**: magnitude of the difference/association (practical relevance).
- **Significance vs practical relevance**: a small p-value may not imply a meaningful effect; always consider effect size and domain context.

## Parametric tests
### One-sample t-test
- Question: Is the mean of one group different from a hypothesized value?
- Assumptions: normality of residuals; independent observations.
- Use case: is a band ratio significantly different from a reference value?
- Reporting: “A one-sample t-test indicated the mean ratio differed from X (t=…, p=…).”
- Implementation: use scipy/statsmodels; not directly wrapped in foodspec.

### Two-sample t-test (independent)
- Question: Do two independent groups have different means?
- Assumptions: normality, similar variances (or use Welch’s), independence.
- Use case: compare ratio means between oil A vs oil B.
- Reporting: “An independent t-test showed a significant difference between oils (t=…, p=…).”
- Implementation: external (SciPy); foodspec outputs ratios/metrics you can feed to tests.

**Example (SciPy)**
```python
import pandas as pd
from scipy.stats import ttest_ind

# df has columns: ratio_1655_1745, oil_type
group_a = df.loc[df["oil_type"] == "olive", "ratio_1655_1745"]
group_b = df.loc[df["oil_type"] == "sunflower", "ratio_1655_1745"]
stat, p = ttest_ind(group_a, group_b, equal_var=False)
print(f"t={stat:.3f}, p={p:.3g}")
```
Interpretation: compares mean band ratios between two oil types. If p is small, the difference is unlikely by chance; consider effect size (e.g., Cohen’s d) for practical relevance.

### Paired t-test
- Question: Do paired measurements differ (e.g., before/after heating)?
- Assumptions: differences are roughly normal; pairs matched.
- Use case: spectra before vs after a treatment on the same sample.
- Reporting: “A paired t-test indicated a change in ratio after heating (t=…, p=…).”
- Implementation: external; foodspec provides features over time.

**Example (SciPy)**
```python
import pandas as pd
from scipy.stats import ttest_rel

# df has columns: ratio_before, ratio_after for matched samples
stat, p = ttest_rel(df["ratio_before"], df["ratio_after"])
print(f"paired t={stat:.3f}, p={p:.3g}")
```
Interpretation: tests mean difference within paired samples (e.g., pre/post heating). Small p suggests a systematic change; report effect size if possible.

### One-way ANOVA
- Question: Do three or more group means differ?
- Assumptions: normality of residuals, homogeneity of variance, independence.
- Use case: compare multiple oil types or treatment stages.
- Reporting: “One-way ANOVA showed a group effect on ratio_1655_1742 (F=…, p=…).”
- Implementation: foodspec’s heating workflow can compute ANOVA; otherwise use SciPy/statsmodels.

**Example (SciPy)**
```python
import pandas as pd
from scipy.stats import f_oneway

# df has columns: ratio_1655_1745, oil_type
groups = [g["ratio_1655_1745"].values for _, g in df.groupby("oil_type")]
stat, p = f_oneway(*groups)
print(f"F={stat:.3f}, p={p:.3g}")
```
Interpretation: tests if at least one oil type differs in mean ratio. If p is small, follow with post-hoc tests and report effect size (e.g., η²).

### MANOVA (brief)
- Question: Do groups differ across multiple dependent variables simultaneously?
- Assumptions: multivariate normality, equal covariance matrices, independence.
- Use case: multiple ratios/PC scores across oil types.
- Reporting: “MANOVA indicated multivariate differences among oils (Wilks’ Λ=…, p=…).”
- Implementation: external; not wrapped in foodspec.

## Non-parametric tests
### Mann–Whitney U
- Question: Do two independent groups differ in central tendency without normality?
- Assumptions: independent samples; ordinal/continuous data.
- Use case: small-sample band ratios with non-normal distributions.
- Reporting: “Mann–Whitney U test found a difference in ratios between groups (U=…, p=…).”

### Kruskal–Wallis
- Question: Are there differences among three or more groups (non-parametric ANOVA)?
- Assumptions: independent samples; similar-shaped distributions.
- Use case: compare ratios across several oils when normality is doubtful.
- Reporting: “Kruskal–Wallis test indicated a group effect (H=…, p=…).”

**Example (SciPy)**
```python
import pandas as pd
from scipy.stats import kruskal

# df has columns: ratio_1655_1745, oil_type
groups = [g["ratio_1655_1745"].values for _, g in df.groupby("oil_type")]
stat, p = kruskal(*groups)
print(f"H={stat:.3f}, p={p:.3g}")
```
Interpretation: tests median differences without assuming normality. Small p suggests at least one group differs; consider post-hoc pairwise tests with corrections.

### Wilcoxon signed-rank
- Question: Paired comparison without normality.
- Use case: before/after heating on the same samples.
- Reporting: “Wilcoxon signed-rank test showed a shift after treatment (W=…, p=…).”

### Friedman test
- Question: Repeated measures across >2 conditions (non-parametric).
- Use case: multiple heating cycles on the same samples.
- Reporting: “Friedman test detected differences across cycles (χ²=…, p=…).”

## Post-hoc & multiple comparisons
- **Tukey HSD**: parametric pairwise group comparisons after ANOVA.
- **Bonferroni/FDR**: adjust p-values when doing many comparisons.
- Reporting: “Post-hoc Tukey tests (adjusted p) identified differences between A and B.”
- Implementation: external; apply to foodspec-generated ratio/features as needed.

## Correlation and regression
- **Pearson correlation**: linear association (assumes normality of variables).
- **Spearman correlation**: rank-based, non-parametric.
- **Simple linear regression**: e.g., ratio vs heating time, slope indicates trend.
- Reporting: “Pearson r=…, p=… between ratio and heating time”; “Slope b=… (95% CI …)”.
- Implementation: foodspec heating workflow fits simple trends; additional correlations via SciPy/pandas.

**Pearson correlation example (SciPy)**
```python
import pandas as pd
from scipy.stats import pearsonr

# df has columns: ratio_1655_1745, heating_time
stat, p = pearsonr(df["ratio_1655_1745"], df["heating_time"])
print(f"r={stat:.3f}, p={p:.3g}")
```
Interpretation: measures linear association between a ratio and time/temperature. Small p suggests a significant linear relationship; r indicates strength/direction.

**Simple regression example (SciPy)**
```python
from scipy.stats import linregress

result = linregress(df["heating_time"], df["ratio_1655_1745"])
print(f"slope={result.slope:.3f}, R²={result.rvalue**2:.3f}, p={result.pvalue:.3g}")
```
Interpretation: slope sign/magnitude shows trend; p tests if slope differs from zero; R² indicates variance explained.

## Effect sizes
- **Cohen’s d**: standardized mean difference (two groups).
- **η² / partial η²**: proportion of variance explained in ANOVA.
- **R²**: proportion of variance explained in regression.
- Reporting: include effect size alongside p-value for practical relevance.

See also
- [metrics_interpretation.md](metrics_interpretation.md)
- [Oil authentication tutorial](../oil_auth_tutorial.md)
- [MethodsX protocol](../protocols/methodsx_protocol.md)
- [API index](../api/index.md)
# Statistical tests (legacy)

> **Status:** Legacy/archived. See the Statistics section (`docs/stats/*`) for the current, maintained content on hypothesis testing, ANOVA, and nonparametric methods.

> **Status: Legacy / Archived**  
> Replaced by the Stats chapters (overview, hypothesis testing, ANOVA/MANOVA, nonparametric, effect sizes). See [Stats overview](../stats/overview.md).
