"""
FoodSpec Visualization Module Documentation
=============================================

Overview
--------
The visualization module provides publication-quality plotting capabilities for 
FoodSpec workflows with strict adherence to reproducibility, artifact management, 
and metadata-aware visualization.

Key Principles
--------------
1. **Reproducibility**: All plots are deterministically seeded for identical output
2. **Artifact Management**: All plots auto-save to ArtifactRegistry.plots_dir
3. **Batch Mode**: Matplotlib Agg backend (no GUI) for headless/cluster execution
4. **Publication Quality**: ≥300 dpi export by default for journal submissions
5. **Metadata Awareness**: Support for batch/stage/instrument grouping and coloring
6. **Figure Returns**: All functions return matplotlib.figure.Figure objects
7. **Standardization**: Unified titles, subtitles (protocol hash, run_id), legends

Module Structure
----------------

foodspec/viz/
├── __init__.py              # Public API exports
├── plots.py                 # Legacy utilities (compatibility)
└── plots_v2.py             # Modern plotting infrastructure
    ├── PlotConfig           # Configuration dataclass
    ├── _init_plot()         # Figure initialization helper
    ├── _standardize_figure()# Standard formatting
    └── 7 plotting functions # confmat, calibration, features, metrics, etc.

PlotConfig Class
----------------

Controls all plotting behavior:

    from foodspec.viz import PlotConfig
    
    config = PlotConfig(
        dpi=300,              # ≥300 for publication (default: 300)
        figure_size=(12, 6),  # Width, height in inches
        font_size=10,         # All text sizes
        seed=42,              # Reproducibility seed
    )

All plotting functions accept optional `config` parameter.

Plotting Functions
------------------

1. **plot_confusion_matrix(y_true, y_pred, ...)**
   
   Heatmap with annotations for classification results.
   
   Parameters:
       y_true : array-like
           True labels
       y_pred : array-like
           Predicted labels
       class_names : list, optional
           Class label names
       artifacts : ArtifactRegistry, optional
           For auto-saving
       filename : str
           Output filename (default: 'confusion_matrix.png')
       protocol_hash : str, optional
           For subtitle
       run_id : str, optional
           For subtitle
       config : PlotConfig, optional
   
   Returns:
       matplotlib.figure.Figure
   
   Example:
       fig = plot_confusion_matrix(
           y_true, y_pred,
           class_names=['Class 0', 'Class 1', 'Class 2'],
           artifacts=artifacts,
           protocol_hash="abc123",
           run_id="run_001",
       )


2. **plot_calibration_curve(y_true, proba, ...)**
   
   Reliability diagram showing probability calibration.
   
   Parameters:
       y_true : array-like
           True labels
       proba : array-like, shape (n_samples, n_classes)
           Predicted probabilities (e.g., from model.predict_proba)
       n_bins : int
           Number of probability bins (default: 10)
       metadata_df : DataFrame, optional
           For coloring by batch/stage/instrument
       metadata_col : str, optional
           Column name for grouping (e.g., 'batch_id')
       artifacts : ArtifactRegistry, optional
       filename : str
       protocol_hash : str, optional
       run_id : str, optional
       config : PlotConfig, optional
   
   Returns:
       matplotlib.figure.Figure
   
   Example:
       fig = plot_calibration_curve(
           y_true, proba,
           n_bins=10,
           metadata_df=metadata,
           metadata_col='batch_id',  # Color by batch
           artifacts=artifacts,
       )


3. **plot_feature_importance(importance_df, ...)**
   
   Horizontal bar chart of top-k features.
   
   Parameters:
       importance_df : DataFrame
           Must have 'feature' and 'importance' columns
       top_k : int
           Number of top features to display (default: 15)
       metadata_group : str, optional
           Group name for subtitle
       artifacts : ArtifactRegistry, optional
       filename : str
       protocol_hash : str, optional
       run_id : str, optional
       config : PlotConfig, optional
   
   Returns:
       matplotlib.figure.Figure
   
   Example:
       importance_df = pd.DataFrame({
           'feature': ['Feature_0', 'Feature_1', ...],
           'importance': [0.42, 0.38, ...],
       })
       fig = plot_feature_importance(importance_df, top_k=10)


4. **plot_metrics_by_fold(metrics_dict, fold_ids, ...)**
   
   Box/violin plot comparing metrics across CV folds.
   
   Parameters:
       metrics_dict : dict
           {'metric_name': [values per fold]}
       fold_ids : array-like
           Fold indices for each metric value
       metadata_df : DataFrame, optional
           For grouping
       metadata_col : str, optional
           Column for grouping
       artifacts : ArtifactRegistry, optional
       filename : str
       protocol_hash : str, optional
       run_id : str, optional
       config : PlotConfig, optional
   
   Returns:
       matplotlib.figure.Figure
   
   Example:
       metrics = {
           'accuracy': [0.87, 0.85, 0.88],
           'f1': [0.84, 0.82, 0.85],
       }
       fig = plot_metrics_by_fold(metrics, fold_ids=[0, 1, 2])


5. **plot_conformal_coverage_by_group(coverage_df, ...)**
   
   Bar chart with error bars for conformal prediction coverage.
   
   Parameters:
       coverage_df : DataFrame
           Must have 'group', 'coverage', 'ci_lower', 'ci_upper'
       metric_col : str
           Column for metric values (default: 'coverage')
       artifacts : ArtifactRegistry, optional
       filename : str
       protocol_hash : str, optional
       run_id : str, optional
       config : PlotConfig, optional
   
   Returns:
       matplotlib.figure.Figure
   
   Example:
       coverage = pd.DataFrame({
           'group': ['Stage 0', 'Stage 1'],
           'coverage': [0.89, 0.91],
           'ci_lower': [0.84, 0.86],
           'ci_upper': [0.94, 0.96],
       })
       fig = plot_conformal_coverage_by_group(coverage)


6. **plot_abstention_rate(abstention_df, ...)**
   
   Distribution of abstention rates across groups.
   
   Parameters:
       abstention_df : DataFrame
           Summary with columns: group, batch, stage, abstention_rate, count
       artifacts : ArtifactRegistry, optional
       filename : str
       protocol_hash : str, optional
       run_id : str, optional
       config : PlotConfig, optional
   
   Returns:
       matplotlib.figure.Figure
   
   Example:
       abstention = pd.DataFrame({
           'batch': [0, 1, 2],
           'abstention_rate': [0.05, 0.08, 0.04],
       })
       fig = plot_abstention_rate(abstention)


7. **plot_prediction_set_sizes(set_sizes_df, ...)**
   
   Histogram of conformal prediction set sizes.
   
   Parameters:
       set_sizes_df : DataFrame
           Must have columns with set size information
       artifacts : ArtifactRegistry, optional
       filename : str
       protocol_hash : str, optional
       run_id : str, optional
       config : PlotConfig, optional
   
   Returns:
       matplotlib.figure.Figure
   
   Example:
       set_sizes = pd.DataFrame({
           'set_size': [1, 1, 2, 1, 3, 2, ...],
       })
       fig = plot_prediction_set_sizes(set_sizes)

Usage Patterns
--------------

1. **Basic Usage (No Artifacts)**

    from foodspec.viz import plot_confusion_matrix
    import matplotlib.pyplot as plt
    
    fig = plot_confusion_matrix(y_true, y_pred)
    fig.show()


2. **With Artifact Saving**

    from foodspec.core.artifacts import ArtifactRegistry
    
    artifacts = ArtifactRegistry(Path('/tmp/output'))
    fig = plot_confusion_matrix(
        y_true, y_pred,
        artifacts=artifacts,
        filename='my_confusion_matrix.png'
    )
    # Automatically saved to /tmp/output/plots/my_confusion_matrix.png


3. **With Metadata Coloring**

    metadata = pd.DataFrame({
        'batch_id': [0, 0, 1, 1, 2, 2],
        'stage': [0, 1, 0, 1, 0, 1],
    })
    
    fig = plot_calibration_curve(
        y_true, proba,
        metadata_df=metadata,
        metadata_col='batch_id',  # Color by batch
        artifacts=artifacts,
    )


4. **Publication Quality**

    from foodspec.viz import PlotConfig
    
    config = PlotConfig(
        dpi=300,           # Journal requirement
        figure_size=(8, 6),
        font_size=11,
        seed=42,           # Reproducible
    )
    
    fig = plot_confusion_matrix(
        y_true, y_pred,
        config=config,
        artifacts=artifacts,
        protocol_hash="abc123",
        run_id="run_001_submission",
    )


5. **Reproducible Batch Processing**

    # Same seed -> identical plots (pixel-perfect reproducibility)
    config = PlotConfig(dpi=300, seed=42)
    
    for fold in range(3):
        fig = plot_feature_importance(
            importance_df,
            config=config,
            artifacts=artifacts,
            filename=f'features_fold_{fold}.png',
        )


Integration with Orchestrator
------------------------------

The orchestrator will integrate visualization as follows:

    from foodspec.core.orchestrator import ExecutionEngine
    from foodspec.viz import plot_confusion_matrix, PlotConfig
    
    engine = ExecutionEngine()
    result = engine.run(protocol, outdir, seed=42)
    
    # Visualization automatically generated based on protocol.visualization.plots
    # All plots saved to result.artifacts.plots_dir


Constraint Compliance
---------------------

✓ **Reproducibility**: Seeded randomness (np.random.seed)
✓ **Artifact Registry**: Auto-saves to artifacts.plots_dir with metadata
✓ **Batch Mode**: Matplotlib Agg backend (no GUI dependencies)
✓ **High-DPI**: 300 dpi default (configurable ≥300)
✓ **Metadata Support**: batch/stage/instrument grouping and coloring
✓ **Figure Returns**: All functions return matplotlib.figure.Figure
✓ **Standardization**: Unified titles, subtitles, legends, tight layout

Testing
-------

Comprehensive test suite in tests/test_viz_comprehensive.py (29 tests):

- PlotConfig validation
- All 7 plotting functions
- Reproducibility (seed consistency)
- Artifact saving (paths, file existence, DPI)
- Metadata coloring (batch/stage grouping)
- Error handling (invalid inputs)

Run tests:
    pytest tests/test_viz_comprehensive.py -v


Performance Considerations
---------------------------

- All functions are stateless (no caching)
- Seeding overhead is minimal (~1ms per plot)
- DPI does not significantly impact performance (export time <1s typical)
- Large datasets (1000+ samples) may require figure_size adjustment
- Metadata coloring scales linearly with group count

Future Extensions
-----------------

1. Custom plot templates for domain-specific visualizations
2. Interactive HTML plots (Plotly integration)
3. Multi-panel layouts for comprehensive reports
4. Animation support for time-series analysis
5. 3D visualization for high-dimensional data

References
----------

- Matplotlib Documentation: https://matplotlib.org/
- ArtifactRegistry: foodspec.core.artifacts
- PlotConfig: foodspec.viz.plots_v2
- Example Notebook: examples/notebooks/trust_visualization_workflow.ipynb
"""
