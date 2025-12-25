# Statistics & Data Analysis Overview

This page summarizes the statistical tools available in FoodSpec and where to use them. Use it as a quick map from common questions to recommended methods, functions, and workflows.

> For notation and symbols, see the [Glossary](../09-reference/glossary.md).

## Questions → Approaches
| Question | Recommended approach | Functions | Example workflow |
| --- | --- | --- | --- |
| Are two groups different (e.g., reference vs suspect)? | Two-sample or paired t-test | `run_ttest` | Batch QC |
| Do ≥3 groups differ (e.g., oil types, heating stages)? | One-way ANOVA (+ Tukey) | `run_anova`, `run_tukey_hsd` | Oil authentication, Heating |
| Are multivariate features different across groups? | MANOVA (if statsmodels installed) | `run_manova` | Multivariate oil/microbial features |
| Which groups differ pairwise? | Post-hoc comparisons | `run_tukey_hsd` | Oil authentication |
| How large is the difference? | Effect sizes | `compute_cohens_d`, `compute_anova_effect_sizes` | Any comparative study |
| Are two variables associated? | Correlation (Pearson/Spearman) | `compute_correlations`, `compute_correlation_matrix` | Heating ratios vs time/quality |
| Is there a lagged relationship? | Cross-correlation for sequences | `compute_cross_correlation` | Time-resolved heating/processing |
| Is my design sufficient? | Design checks | `summarize_group_sizes`, `check_minimum_samples` | All workflows |

## How to explore
- Theory chapters: see the stats section in the nav (hypothesis testing, ANOVA/MANOVA, correlations, design).
- API reference: [Statistics](../08-api/stats.md).
- Workflows: each workflow page includes a “Statistical analysis” section with code snippets and interpretation.
- Protocols: reproducibility checklist and benchmarking framework explain how to report and compare results.
---

## When Results Cannot Be Trusted

⚠️ **Red flags invalidating statistical conclusions:**

1. **Test chosen without examining assumptions**
   - Parametric tests (t-test, ANOVA) require normality and equal variances
   - Applying parametric tests to non-normal data inflates Type I error
   - **Fix:** Check Q-Q plots or Shapiro-Wilk test; use nonparametric alternatives if violated

2. **P-value without checking actual difference magnitude**
   - Significance (p < 0.05) ≠ practical importance
   - Large sample sizes can detect tiny, irrelevant differences; small samples miss real effects
   - **Fix:** Report effect sizes (Cohen's d, R²) alongside p-values; discuss practical relevance

3. **Cherry-picked post-hoc comparisons (report only 3 out of 20 comparisons)**
   - Selective reporting inflates false positive rates
   - P-value interpretation assumes pre-specified hypotheses, not post-hoc hunting
   - **Fix:** Declare comparisons of interest before analysis; apply multiple-testing correction

4. **No correction for multiple tests (testing 200 peaks, reporting p < 0.05 without adjustment)**
   - 200 independent tests expect ~10 false positives at α=0.05
   - Uncorrected p-values are misleading
   - **Fix:** Use Bonferroni, Benjamini–Hochberg FDR, or permutation tests; report adjusted p-values

5. **Outliers removed without justification**
   - Removing extreme values to "improve" results is data fabrication
   - Real spectroscopy and natural products have variability; removing it hides real uncertainty
   - **Fix:** Identify outliers a priori based on QC criteria; document and report removals

6. **Comparing groups of very different sizes (n=100 vs n=3)**
   - Statistical power dominated by smallest group
   - Heterogeneous variance assumptions violated
   - **Fix:** Aim for similar group sizes; if imbalanced, use Welch's test, rank tests, or permutation tests

7. **Independence assumption violated (20 technical replicates of 1 sample ≠ 20 independent samples)**
   - Repeated measurements of same unit are autocorrelated, not independent
   - Tests assuming independence produce inflated significance
   - **Fix:** Analyze distinct samples as replicates; document which measurements are paired or nested

8. **Batch confounding (treatment A all from Day 1, B all from Day 2)**
   - Systematic drift (instrument, temperature, operator) accumulates with batch, not treatment
   - Impossible to disentangle batch from treatment effects
   - **Fix:** Randomize sample order; use batch-aware analysis (include batch in model or GroupKFold CV)

---
## Notes
- Ensure preprocessing and wavenumber alignment are consistent before running tests.
- Check assumptions (normality, variance, independence) and consider nonparametric options when violated.
- Report p-values and effect sizes; discuss practical (food science) relevance alongside statistical significance.
