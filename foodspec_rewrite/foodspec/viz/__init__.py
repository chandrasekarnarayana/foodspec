"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
Visualization module: Spectral plots and diagnostics.

Visualizing spectral data and model results:
    from foodspec.viz import plot_raw_vs_processed, plot_pca_scatter
    plot_raw_vs_processed(wavenumbers, raw, processed, artifacts)
    plot_pca_scatter(scores, labels, artifacts)
"""

from foodspec.viz.plots import (
    plot_heatmap,
    plot_pca_scatter,
    plot_raw_vs_processed,
    plot_reliability_diagram,
)
from foodspec.viz.plots_v2 import (
    PlotConfig,
    plot_abstention_rate,
    plot_calibration_curve,
    plot_conformal_coverage_by_group,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_by_fold,
    plot_prediction_set_sizes,
)
from foodspec.viz.pipeline import (
    get_pipeline_stats,
    plot_pipeline_dag,
)
from foodspec.viz.parameters import (
    get_parameter_summary,
    plot_parameter_map,
)
from foodspec.viz.lineage import (
    get_lineage_summary,
    plot_data_lineage,
)
from foodspec.viz.badges import (
    plot_reproducibility_badge,
    get_reproducibility_status,
)
from foodspec.viz.drift import (
    plot_batch_drift,
    plot_stage_differences,
    get_batch_statistics,
    get_stage_statistics,
    plot_replicate_similarity,
    plot_temporal_drift,
)
from foodspec.viz.interpretability import (
    plot_importance_overlay,
    plot_marker_bands,
    get_band_statistics,
)
from foodspec.viz.uncertainty import (
    plot_confidence_map,
    plot_set_size_distribution,
    plot_coverage_efficiency,
    plot_abstention_distribution,
    get_uncertainty_statistics,
)
from foodspec.viz.processing_stages import (
    plot_processing_stages,
    plot_preprocessing_comparison,
    get_processing_statistics,
)
from foodspec.viz.embeddings import (
    plot_embedding,
    plot_embedding_comparison,
    get_embedding_statistics,
)
from foodspec.viz.paper_styles import apply_paper_style, PAPER_STYLES, DEFAULT_DPI
from foodspec.viz.paper import (
    FigurePreset,
    apply_figure_preset,
    figure_context,
    save_figure,
    get_figure_preset_config,
    list_presets,
)
from foodspec.viz.visualization_manager import (
    VisualizationManager,
    run_all_visualizations,
)

__all__ = [
    "PlotConfig",
    "plot_confusion_matrix",
    "plot_heatmap",
    "plot_pca_scatter",
    "plot_raw_vs_processed",
    "plot_reliability_diagram",
    "plot_calibration_curve",
    "plot_metrics_by_fold",
    "plot_feature_importance",
    "plot_conformal_coverage_by_group",
    "plot_prediction_set_sizes",
    "plot_pipeline_dag",
    "get_pipeline_stats",
    "plot_parameter_map",
    "get_parameter_summary",
    "plot_data_lineage",
    "get_lineage_summary",
    "plot_abstention_rate",
    "plot_reproducibility_badge",
    "get_reproducibility_status",
    "plot_batch_drift",
    "plot_stage_differences",
    "get_batch_statistics",
    "get_stage_statistics",
    "plot_replicate_similarity",
    "plot_temporal_drift",
    "plot_importance_overlay",
    "plot_marker_bands",
    "get_band_statistics",
    "plot_confidence_map",
    "plot_set_size_distribution",
    "plot_coverage_efficiency",
    "plot_abstention_distribution",
    "get_uncertainty_statistics",
    "plot_processing_stages",
    "plot_preprocessing_comparison",
    "get_processing_statistics",
    "plot_embedding",
    "plot_embedding_comparison",
    "get_embedding_statistics",
    "apply_paper_style",
    "PAPER_STYLES",
    "DEFAULT_DPI",
    "FigurePreset",
    "apply_figure_preset",
    "figure_context",
    "save_figure",
    "get_figure_preset_config",
    "list_presets",
    "VisualizationManager",
    "run_all_visualizations",
]

