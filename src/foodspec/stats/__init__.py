"""
Statistical utilities for FoodSpec.

This subpackage wraps common hypothesis tests, correlation analyses, and effect
size calculations with simple interfaces that accept NumPy/pandas inputs and
FoodSpectrumSet metadata. Use these helpers to quantify differences between
groups (e.g., oil types), assess correlations (e.g., ratios vs heating time),
and summarize study design balance.
"""

from foodspec.stats.correlations import (
    compute_correlation_matrix,
    compute_correlations,
    compute_cross_correlation,
)
from foodspec.stats.design import check_minimum_samples, summarize_group_sizes
from foodspec.stats.effects import compute_anova_effect_sizes, compute_cohens_d
from foodspec.stats.hypothesis_tests import (
    games_howell,
    run_anova,
    run_friedman_test,
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_manova,
    run_ttest,
    run_tukey_hsd,
    run_wilcoxon_signed_rank,
)
from foodspec.stats.hypothesis_tests import run_shapiro, benjamini_hochberg
from foodspec.stats.robustness import bootstrap_metric, permutation_test_metric
from foodspec.stats.reporting import stats_report_for_feature, stats_report_for_features_table
from foodspec.stats.distances import (
     euclidean_distance,
     cosine_distance,
     pearson_distance,
     sid_distance,
     sam_angle,
     compute_distances,
)
from foodspec.stats.time_metrics import (
    linear_slope,
    quadratic_acceleration,
    rolling_slope,
)
from foodspec.stats.fusion_metrics import (
    modality_agreement_kappa,
    modality_consistency_rate,
    cross_modality_correlation,
)

__all__ = [
    "run_ttest",
    "run_anova",
    "run_manova",
    "run_tukey_hsd",
    "games_howell",
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
     # fusion metrics
     "modality_agreement_kappa",
     "modality_consistency_rate",
     "cross_modality_correlation",
]
