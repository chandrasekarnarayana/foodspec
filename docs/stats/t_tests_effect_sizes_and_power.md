# t-Tests, Effect Sizes, and Power

This chapter revisits t-tests for two-group comparisons, introduces effect sizes, and discusses power qualitatively for food spectroscopy.

## t-tests recap
- **One-sample:** Compare a mean to a reference \( \mu_0 \).
- **Two-sample (Welch):** Compare two independent group means.
- **Paired:** Compare matched pairs (before/after heating).
- Statistic (two-sample, Welch): \( t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}} \)

## Effect sizes
- **Cohen’s d:** Standardized mean difference.
  - Pooled SD: \( d = (\bar{x}_1 - \bar{x}_2)/s_p \).
  - Interpret alongside p-values to gauge magnitude.
- **Confidence intervals:** Not computed directly in FoodSpec; use bootstrap resampling (e.g., `bootstrap_metric` on Cohen’s d) to obtain empirical CIs, or rely on external stats libraries if analytic CIs are required.

## Power (qualitative)
- Influenced by effect size, variance, sample size, and significance level.
- More replicates and lower noise increase power; good preprocessing reduces variance.
- For small datasets, lack of significance may reflect low power rather than no effect.

## Food spectroscopy examples
- Compare peak ratio between authentic vs suspect batches (two-sample).
- Paired comparison before/after mild heating of the same oil.

## Code snippet
```python
from foodspec.stats import run_ttest, compute_cohens_d

g1 = [1.0, 1.1, 0.9]
g2 = [1.8, 1.9, 1.7]
res = run_ttest(g1, g2)
print(res.summary)
print("Cohen's d:", compute_cohens_d(g1, g2))
```

## Interpretation
- Report t-statistic, df, p-value, and effect size. Discuss practical importance (e.g., does d correspond to meaningful quality change?).
- Avoid equating non-significance with equivalence; consider power and confidence intervals.

## Further reading
- [Hypothesis testing](hypothesis_testing_in_food_spectroscopy.md)
- [ANOVA and MANOVA](anova_and_manova.md)
- [Study design](study_design_and_data_requirements.md)
- API: [Statistics](../api/stats.md)
