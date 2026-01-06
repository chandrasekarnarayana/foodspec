# Statistics API

Statistical testing and effect size calculations for spectral analysis.

The `foodspec.stats` module provides hypothesis testing, correlation analysis, and statistical reporting tools for comparing spectral measurements.

## Hypothesis Testing

### run_ttest

Independent or paired t-test.

::: foodspec.stats.hypothesis_tests.run_ttest
    options:
      show_source: false
      heading_level: 4

### run_anova

One-way ANOVA for comparing multiple groups.

::: foodspec.stats.hypothesis_tests.run_anova
    options:
      show_source: false
      heading_level: 4

### run_manova

Multivariate ANOVA for multiple dependent variables.

::: foodspec.stats.hypothesis_tests.run_manova
    options:
      show_source: false
      heading_level: 4

### run_kruskal_wallis

Non-parametric alternative to ANOVA.

::: foodspec.stats.hypothesis_tests.run_kruskal_wallis
    options:
      show_source: false
      heading_level: 4

### run_mannwhitney_u

Non-parametric alternative to t-test.

::: foodspec.stats.hypothesis_tests.run_mannwhitney_u
    options:
      show_source: false
      heading_level: 4

## Effect Sizes

### compute_cohens_d

Cohen's d effect size for group comparisons.

::: foodspec.stats.effects.compute_cohens_d
    options:
      show_source: false
      heading_level: 4

### compute_anova_effect_sizes

Effect sizes (eta-squared, omega-squared) for ANOVA.

::: foodspec.stats.effects.compute_anova_effect_sizes
    options:
      show_source: false
      heading_level: 4

## Correlations & Distances

### compute_correlations

Pearson or Spearman correlation between features.

::: foodspec.stats.correlations.compute_correlations
    options:
      show_source: false
      heading_level: 4

### euclidean_distance

Euclidean distance between spectra.

::: foodspec.stats.distances.euclidean_distance
    options:
      show_source: false
      heading_level: 4

### cosine_distance

Cosine distance (1 - cosine similarity).

::: foodspec.stats.distances.cosine_distance
    options:
      show_source: false
      heading_level: 4

## Robustness Analysis

### bootstrap_metric

Bootstrap confidence intervals for metrics.

::: foodspec.stats.robustness.bootstrap_metric
    options:
      show_source: false
      heading_level: 4

### permutation_test_metric

Permutation test for statistical significance.

::: foodspec.stats.robustness.permutation_test_metric
    options:
      show_source: false
      heading_level: 4

## See Also

- **[Statistics Methods](../methods/statistics/introduction_to_statistical_analysis.md)** - Statistical methodology
- **[Study Design](../methods/statistics/study_design_and_data_requirements.md)** - Planning statistical analyses
- **[Examples](../examples_gallery.md)** - Statistical analysis workflows
