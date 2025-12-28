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

---

## When Results Cannot Be Trusted

⚠️ **Red flags for t-test and effect size reliability:**

1. **Small sample size (n < 5 per group) with no power analysis**
   - t-test power depends critically on n; underpowered tests miss real effects and inflat false negatives
   - Effects sizes from tiny samples (d) are unstable and don't generalize
   - **Fix:** Conduct power analysis before study; report target n based on effect size of interest

2. **Huge effect size (d > 3) without mechanistic explanation**
   - t-test can produce unrealistic d when n is small and variance is low
   - May indicate group non-overlap, measurement error, or data entry mistakes
   - **Fix:** Visualize distributions; inspect for outliers or data errors; explain large effect biologically

3. **Non-normal distributions (Q-Q plot shows fat tails or skew) with parametric t-test**
   - t-tests assume normality; violating this inflates Type I error (false positives)
   - Heavy tails or skew make p-values unreliable
   - **Fix:** Check normality; use Mann–Whitney U test (nonparametric alternative) if violated

4. **Unequal variances (SD_A = 0.1, SD_B = 1.0) without Welch's t-test**
   - Standard Student's t-test assumes equal variances; violation inflates Type I error
   - Welch's t-test is more robust when variances differ
   - **Fix:** Check variance homogeneity (Levene's test); use Welch's t-test by default

5. **Comparing pairs that aren't actually paired**
   - Paired t-tests require genuine pairing (e.g., same sample before/after)
   - False pairing artificially reduces variance and inflates significance
   - **Fix:** Verify pairing structure; use unpaired test if no true pairing

6. **Cherry-picking which groups to compare (reporting only significant t-tests)**
   - If you perform multiple t-tests without pre-registration, p-values are inflated
   - Each test has 5% false positive rate; testing 20 pairs expects 1 false positive
   - **Fix:** Pre-register comparisons; apply multiple-testing correction

7. **Interpreting non-significance (p > 0.05) as proof of equality**
   - Failure to reject H₀ ≠ proof that groups are equal
   - Power may be too low to detect real differences
   - **Fix:** Report confidence intervals for the difference; compute post-hoc power

8. **Effect size from same dataset used for statistical test**
   - Reporting d from the same data that gave significant p-value produces inflated d
   - Effect sizes shrink in independent validation (publication bias)
   - **Fix:** Report confidence intervals around d; plan independent validation

## Further reading
- [Hypothesis testing](hypothesis_testing_in_food_spectroscopy.md)
- [ANOVA and MANOVA](anova_and_manova.md)
- [Study design](study_design_and_data_requirements.md)
- API: [Statistics](../../api/stats.md)
