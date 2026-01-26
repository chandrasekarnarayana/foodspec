# Statistics API

Statistical testing and effect size calculations for spectral analysis.

The `foodspec.stats` module provides hypothesis testing, correlation analysis, and statistical reporting tools for comparing spectral measurements.

## Time Metrics

Functions for analyzing temporal degradation trends.

### linear_slope

Linear trend slope and intercept for time series.

::: foodspec.stats.time_metrics.linear_slope
    options:
      show_source: false
      heading_level: 4

### quadratic_acceleration

Acceleration coefficient from a quadratic fit to time series.

::: foodspec.stats.time_metrics.quadratic_acceleration
    options:
      show_source: false
      heading_level: 4

### rolling_slope

Rolling-window linear slope across time series.

::: foodspec.stats.time_metrics.rolling_slope
    options:
      show_source: false
      heading_level: 4

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

### run_anderson_darling

Anderson-Darling normality test.

::: foodspec.stats.hypothesis_tests.run_anderson_darling
    options:
      show_source: false
      heading_level: 4

### run_levene

Levene test for equal variances.

::: foodspec.stats.hypothesis_tests.run_levene
    options:
      show_source: false
      heading_level: 4

### run_bartlett

Bartlett test for equal variances (normality assumed).

::: foodspec.stats.hypothesis_tests.run_bartlett
    options:
      show_source: false
      heading_level: 4

### run_ancova

ANCOVA with group factor + covariates.

::: foodspec.stats.hypothesis_tests.run_ancova
    options:
      show_source: false
      heading_level: 4

### run_tost_equivalence

Equivalence testing using TOST.

::: foodspec.stats.hypothesis_tests.run_tost_equivalence
    options:
      show_source: false
      heading_level: 4

### run_noninferiority

Noninferiority testing against a margin.

::: foodspec.stats.hypothesis_tests.run_noninferiority
    options:
      show_source: false
      heading_level: 4

### group_sequential_boundaries

Group sequential boundary calculator.

::: foodspec.stats.hypothesis_tests.group_sequential_boundaries
    options:
      show_source: false
      heading_level: 4

### check_group_sequential

Evaluate sequential z-statistics against boundaries.

::: foodspec.stats.hypothesis_tests.check_group_sequential
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

## Clustering

### kmeans_cluster

K-means clustering with diagnostics.

::: foodspec.stats.clustering.kmeans_cluster
    options:
      show_source: false
      heading_level: 4

### hierarchical_cluster

Hierarchical clustering with linkage output.

::: foodspec.stats.clustering.hierarchical_cluster
    options:
      show_source: false
      heading_level: 4

### fuzzy_c_means

Fuzzy clustering for soft assignments.

::: foodspec.stats.clustering.fuzzy_c_means
    options:
      show_source: false
      heading_level: 4

## Distribution Fitting

### fit_distribution

Fit common distributions (Weibull/Gamma/Beta/Normal).

::: foodspec.stats.distribution_fitting.fit_distribution
    options:
      show_source: false
      heading_level: 4

### probability_plot_data

Probability plot diagnostics.

::: foodspec.stats.distribution_fitting.probability_plot_data
    options:
      show_source: false
      heading_level: 4

## Design of Experiments

### full_factorial_2level

Full factorial 2-level design.

::: foodspec.stats.doe.full_factorial_2level
    options:
      show_source: false
      heading_level: 4

### central_composite_design

Response surface central composite design.

::: foodspec.stats.doe.central_composite_design
    options:
      show_source: false
      heading_level: 4

## Time Series

### fit_arima

ARIMA forecasting.

::: foodspec.stats.time_series.fit_arima
    options:
      show_source: false
      heading_level: 4

### fit_exponential_smoothing

Exponential smoothing forecasting.

::: foodspec.stats.time_series.fit_exponential_smoothing
    options:
      show_source: false
      heading_level: 4

## Method Comparison

### passing_bablok

Passing-Bablok regression.

::: foodspec.stats.method_comparison.passing_bablok
    options:
      show_source: false
      heading_level: 4

### lins_concordance_correlation

Lin's concordance correlation coefficient.

::: foodspec.stats.method_comparison.lins_concordance_correlation
    options:
      show_source: false
      heading_level: 4

## Diagnostics

### adjusted_r2

Adjusted R-squared.

::: foodspec.stats.diagnostics.adjusted_r2
    options:
      show_source: false
      heading_level: 4

### cronbach_alpha

Cronbach's alpha reliability.

::: foodspec.stats.diagnostics.cronbach_alpha
    options:
      show_source: false
      heading_level: 4

### runs_test

Runs test for randomness.

::: foodspec.stats.diagnostics.runs_test
    options:
      show_source: false
      heading_level: 4

### normal_tolerance_interval

Approximate tolerance interval for normal data.

::: foodspec.stats.diagnostics.normal_tolerance_interval
    options:
      show_source: false
      heading_level: 4

## See Also

- **[Statistics Methods](../methods/statistics/introduction_to_statistical_analysis.md)** - Statistical methodology
- **[Study Design](../methods/statistics/study_design_and_data_requirements.md)** - Planning statistical analyses
- **[Examples](../examples_gallery.md)** - Statistical analysis workflows
