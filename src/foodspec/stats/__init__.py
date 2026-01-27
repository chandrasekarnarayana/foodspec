"""
Statistical utilities for FoodSpec.

This subpackage wraps common hypothesis tests, correlation analyses, and effect
size calculations with simple interfaces that accept NumPy/pandas inputs and
FoodSpectrumSet metadata. Use these helpers to quantify differences between
groups (e.g., oil types), assess correlations (e.g., ratios vs heating time),
and summarize study design balance.
"""

from foodspec.stats.clustering import (
    ClusteringResult,
    FuzzyCMeansResult,
    HierarchicalClusteringResult,
    RegressionClusterResult,
    fuzzy_c_means,
    hierarchical_cluster,
    kmeans_cluster,
    regression_clustering,
)
from foodspec.stats.correlations import (
    compute_correlation_matrix,
    compute_correlations,
    compute_cross_correlation,
)
from foodspec.stats.design import check_minimum_samples, summarize_group_sizes
from foodspec.stats.diagnostics import (
    RunsTestResult,
    adjusted_r2,
    aic_from_rss,
    bic_from_rss,
    cronbach_alpha,
    normal_tolerance_interval,
    runs_test,
)
from foodspec.stats.distances import (
    compute_distances,
    cosine_distance,
    euclidean_distance,
    pearson_distance,
    sam_angle,
    sid_distance,
)
from foodspec.stats.distribution_fitting import (
    DistributionFit,
    compare_distributions,
    fit_distribution,
    probability_plot_data,
)
from foodspec.stats.doe import (
    DOptimalResult,
    central_composite_design,
    d_optimal_design,
    fractional_factorial_2level,
    full_factorial_2level,
    randomized_block_design,
)
from foodspec.stats.effects import compute_anova_effect_sizes, compute_cohens_d
from foodspec.stats.fusion_metrics import (
    cross_modality_correlation,
    modality_agreement_kappa,
    modality_consistency_rate,
)
from foodspec.stats.hypothesis_tests import (
    benjamini_hochberg,
    check_group_sequential,
    games_howell,
    group_sequential_boundaries,
    run_ancova,
    run_anderson_darling,
    run_anova,
    run_bartlett,
    run_friedman_test,
    run_kruskal_wallis,
    run_levene,
    run_mannwhitney_u,
    run_manova,
    run_noninferiority,
    run_shapiro,
    run_tost_equivalence,
    run_ttest,
    run_tukey_hsd,
    run_welch_ttest,
    run_wilcoxon_signed_rank,
)
from foodspec.stats.method_comparison import (
    BlandAltmanResult,
    PassingBablokResult,
    bland_altman,
    bland_altman_plot,
    lins_concordance_correlation,
    passing_bablok,
    passing_bablok_plot,
)
from foodspec.stats.reporting import stats_report_for_feature, stats_report_for_features_table
from foodspec.stats.robustness import bootstrap_metric, permutation_test_metric
from foodspec.stats.time_metrics import (
    linear_slope,
    quadratic_acceleration,
    rolling_slope,
)
from foodspec.stats.time_series import (
    ForecastResult,
    autocorrelation,
    cross_correlation,
    fit_arima,
    fit_exponential_smoothing,
    spectral_analysis,
)

__all__ = [
    "run_ttest",
    "run_anova",
    "run_ancova",
    "run_manova",
    "run_tukey_hsd",
    "games_howell",
    "run_anderson_darling",
    "run_levene",
    "run_bartlett",
    "run_welch_ttest",
    "run_tost_equivalence",
    "run_noninferiority",
    "group_sequential_boundaries",
    "check_group_sequential",
    "compute_correlations",
    "compute_correlation_matrix",
    "compute_cross_correlation",
    "compute_cohens_d",
    "compute_anova_effect_sizes",
    "summarize_group_sizes",
    "check_minimum_samples",
    "run_kruskal_wallis",
    "run_mannwhitney_u",
    "run_wilcoxon_signed_rank",
    "run_friedman_test",
    "bootstrap_metric",
    "permutation_test_metric",
    "run_shapiro",
    "benjamini_hochberg",
    "stats_report_for_feature",
    "stats_report_for_features_table",
    # clustering
    "ClusteringResult",
    "HierarchicalClusteringResult",
    "FuzzyCMeansResult",
    "RegressionClusterResult",
    "kmeans_cluster",
    "hierarchical_cluster",
    "fuzzy_c_means",
    "regression_clustering",
    # distribution fitting
    "DistributionFit",
    "fit_distribution",
    "compare_distributions",
    "probability_plot_data",
    # doe
    "DOptimalResult",
    "full_factorial_2level",
    "fractional_factorial_2level",
    "central_composite_design",
    "randomized_block_design",
    "d_optimal_design",
    # distances
    "euclidean_distance",
    "cosine_distance",
    "pearson_distance",
    "sid_distance",
    "sam_angle",
    "compute_distances",
    # time metrics
    "linear_slope",
    "quadratic_acceleration",
    "rolling_slope",
    # time series
    "ForecastResult",
    "autocorrelation",
    "cross_correlation",
    "spectral_analysis",
    "fit_arima",
    "fit_exponential_smoothing",
    # fusion metrics
    "modality_agreement_kappa",
    "modality_consistency_rate",
    "cross_modality_correlation",
    # method comparison
    "BlandAltmanResult",
    "PassingBablokResult",
    "bland_altman",
    "bland_altman_plot",
    "passing_bablok",
    "passing_bablok_plot",
    "lins_concordance_correlation",
    # diagnostics
    "RunsTestResult",
    "adjusted_r2",
    "aic_from_rss",
    "bic_from_rss",
    "cronbach_alpha",
    "runs_test",
    "normal_tolerance_interval",
]
