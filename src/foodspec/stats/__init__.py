"""
Statistical utilities for FoodSpec.

This subpackage wraps common hypothesis tests, correlation analyses, and effect
size calculations with simple interfaces that accept NumPy/pandas inputs and
FoodSpectrumSet metadata. Use these helpers to quantify differences between
groups (e.g., oil types), assess correlations (e.g., ratios vs heating time),
and summarize study design balance.
"""

from foodspec.stats.hypothesis_tests import run_ttest, run_anova, run_manova, run_tukey_hsd
from foodspec.stats.correlations import (
    compute_correlations,
    compute_correlation_matrix,
    compute_cross_correlation,
)
from foodspec.stats.robustness import bootstrap_metric, permutation_test_metric
from foodspec.stats.effects import compute_cohens_d, compute_anova_effect_sizes
from foodspec.stats.design import summarize_group_sizes, check_minimum_samples
from foodspec.stats.hypothesis_tests import (
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_wilcoxon_signed_rank,
    run_friedman_test,
)

__all__ = [
    "run_ttest",
    "run_anova",
    "run_manova",
    "run_tukey_hsd",
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
]
