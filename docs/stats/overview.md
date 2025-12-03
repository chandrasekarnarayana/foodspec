# Statistics & Data Analysis Overview

This page summarizes the statistical tools available in FoodSpec and where to use them. Use it as a quick map from common questions to recommended methods, functions, and workflows.

> For notation and symbols, see the [Glossary](../glossary.md).

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
- API reference: [Statistics](../api/stats.md).
- Workflows: each workflow page includes a “Statistical analysis” section with code snippets and interpretation.
- Protocols: MethodsX mapping, reproducibility checklist, and benchmarking framework explain how to report and compare results.

## Notes
- Ensure preprocessing and wavenumber alignment are consistent before running tests.
- Check assumptions (normality, variance, independence) and consider nonparametric options when violated.
- Report p-values and effect sizes; discuss practical (food science) relevance alongside statistical significance.
