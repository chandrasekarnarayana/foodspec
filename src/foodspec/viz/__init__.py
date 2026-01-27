from .clustering import plot_dendrogram
from .coefficients import (
    get_coefficient_statistics,
    plot_coefficients_heatmap,
)
from .compare import (
    RunSummary,
    compare_runs,
    compute_baseline_deltas,
    create_comparison_dashboard,
    create_leaderboard,
    create_monitoring_plot,
    create_radar_plot,
    load_run_summary,
    scan_runs,
)
from .comprehensive import (
    plot_abstention_distribution as plot_abstention_by_group,
)
from .comprehensive import (
    plot_conformal_set_sizes,
    plot_coverage_efficiency_curve,
    plot_pca_umap,
    plot_raw_vs_processed_overlay,
)
from .control_charts import (
    plot_control_chart,
    plot_control_chart_group,
    plot_cusum,
    plot_ewma,
    plot_pareto,
    plot_runs,
)
from .distribution import plot_probability_plot
from .paper import (
    FigurePreset,
    apply_figure_preset,
    figure_context,
    get_figure_preset_config,
    list_presets,
    save_figure,
)
from .plots import (
    plot_bland_altman,
    plot_calibration_with_ci,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_mean_with_ci,
    plot_pca_loadings,
    plot_pca_scores,
    plot_pr_curve,
    plot_regression_calibration,
    plot_residuals,
    plot_roc_curve,
    plot_spectra_overlay,
)
from .ratios import plot_ratio_by_group, plot_ratio_scatter, plot_ratio_vs_continuous
from .stability import (
    get_stability_statistics,
    plot_feature_stability,
)
from .uncertainty import (
    get_uncertainty_statistics,
    plot_abstention_distribution,
    plot_confidence_map,
    plot_coverage_efficiency,
    plot_set_size_distribution,
)

__all__ = [
    "plot_spectra_overlay",
    "plot_mean_with_ci",
    "plot_pca_scores",
    "plot_pca_loadings",
    "plot_confusion_matrix",
    "plot_correlation_heatmap",
    "plot_regression_calibration",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_residuals",
    "plot_calibration_with_ci",
    "plot_bland_altman",
    "plot_ratio_by_group",
    "plot_ratio_scatter",
    "plot_ratio_vs_continuous",
    "plot_coefficients_heatmap",
    "get_coefficient_statistics",
    "plot_feature_stability",
    "get_stability_statistics",
    "plot_confidence_map",
    "plot_set_size_distribution",
    "plot_coverage_efficiency",
    "plot_abstention_distribution",
    "get_uncertainty_statistics",
    "FigurePreset",
    "apply_figure_preset",
    "figure_context",
    "save_figure",
    "get_figure_preset_config",
    "list_presets",
    "plot_control_chart",
    "plot_control_chart_group",
    "plot_cusum",
    "plot_ewma",
    "plot_pareto",
    "plot_runs",
    "plot_probability_plot",
    "plot_dendrogram",
    "RunSummary",
    "compare_runs",
    "compute_baseline_deltas",
    "create_comparison_dashboard",
    "create_leaderboard",
    "create_monitoring_plot",
    "create_radar_plot",
    "load_run_summary",
    "scan_runs",
    "plot_abstention_by_group",
    "plot_conformal_set_sizes",
    "plot_coverage_efficiency_curve",
    "plot_pca_umap",
    "plot_raw_vs_processed_overlay",
]
