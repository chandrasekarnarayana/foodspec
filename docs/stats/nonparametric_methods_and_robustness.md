# Nonparametric Methods and Robustness

Nonparametric tests and robustness checks help when normality or variance assumptions are doubtful, when sample sizes are small, or when data are skewed/ordinal (e.g., sensory scores). This page summarizes when to use them and how to run them with foodspec.

## When to choose nonparametric methods
- Clear skew/outliers or heteroscedasticity that transformations cannot fix.
- Ordinal or rank-based scores (e.g., sensory panels).
- Very small sample sizes where normality is questionable.
- As a sensitivity check alongside parametric tests.

## Core tests (SciPy-backed wrappers in foodspec.stats)
- **Mann–Whitney U** (`run_mannwhitney_u`): two independent groups, rank-based alternative to two-sample t-test.
- **Wilcoxon signed-rank** (`run_wilcoxon_signed_rank`): paired samples, alternative to paired t-test.
- **Kruskal–Wallis** (`run_kruskal_wallis`): >2 independent groups, alternative to one-way ANOVA.
- **Friedman** (`run_friedman_test`): repeated measures across conditions (nonparametric ANOVA analogue).
- **Games–Howell post-hoc** (`games_howell`): pairwise comparisons robust to unequal variances/sizes after ANOVA/Kruskal–Wallis.

### Example: Mann–Whitney U and Kruskal–Wallis
```python
import pandas as pd
from foodspec.stats import run_mannwhitney_u, run_kruskal_wallis

df = pd.DataFrame({
    "ratio": [1.0, 1.1, 1.2, 3.0, 3.1, 3.2],
    "group": ["olive", "olive", "olive", "sunflower", "sunflower", "sunflower"],
})

u_res = run_mannwhitney_u(df, group_col="group", value_col="ratio")
print(u_res.summary)

kw_res = run_kruskal_wallis(df, group_col="group", value_col="ratio")
print(kw_res.summary)
```

### Example: Wilcoxon signed-rank (paired)
```python
from foodspec.stats import run_wilcoxon_signed_rank

before = [1.0, 1.1, 1.2]
after = [1.5, 1.6, 1.7]
res = run_wilcoxon_signed_rank(before, after)
print(res.summary)
```

### Example: Friedman test (repeated measures)
```python
import numpy as np
from foodspec.stats import run_friedman_test

cond1 = np.array([1.0, 1.1, 1.2])
cond2 = np.array([1.3, 1.4, 1.5])
cond3 = np.array([1.6, 1.7, 1.8])
res = run_friedman_test(cond1, cond2, cond3)
print(res.summary)
```

## Robustness checks (bootstrap / permutation)
Bootstrap and permutation help assess stability of metrics (e.g., accuracy, RMSE).

```python
import numpy as np
from foodspec.stats import bootstrap_metric, permutation_test_metric

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 1, 0, 0, 0])

def accuracy(a, b):
    return np.mean(a == b)

boot = bootstrap_metric(accuracy, y_true, y_pred, n_bootstrap=500, random_state=0)
print("Observed:", boot["observed"], "CI:", boot["ci"])

perm = permutation_test_metric(accuracy, y_true, y_pred, n_permutations=500, random_state=0)
print("Permutation p-value:", perm["p_value"])
```

Use cases:
- Quantify uncertainty of metrics on small datasets.
- Check whether observed performance could occur by chance (permutation).
- Report CI/p-values alongside metrics in reports.

## Interpretation notes
- Nonparametric tests are less powerful than parametric counterparts; ensure adequate replication.
- Always inspect distributions/boxplots; pair tests with visualizations.
- Robustness checks should accompany headline metrics in methods/results for transparency.

## See also
- [Hypothesis testing](hypothesis_testing_in_food_spectroscopy.md)
- [Study design](study_design_and_data_requirements.md)
- [Metrics & evaluation](../metrics/metrics_and_evaluation.md)
- [Stats API](../api/stats.md)
