# Visualizations Library

FoodSpec provides a consistent plotting API under `foodspec.viz.api` for
report-ready, deterministic figures. Each plot function can save figures
with stable filenames and a JSON sidecar containing description, inputs,
and code version metadata.

## CLI

Generate all plots from a run directory:

```bash
foodspec viz make --run runs/my_run --outdir runs/viz --all
```

Generate a subset:

```bash
foodspec viz make --run runs/my_run --outdir runs/viz --plots "overlay,pca,confusion,reliability"
```

## Plot catalog

| Plot key | Function | When to use |
| --- | --- | --- |
| overlay | `plot_raw_processed_overlay` | Compare preprocessing impact on spectra |
| spectra_heatmap | `plot_spectra_heatmap` | Inspect global spectral patterns |
| correlation | `plot_correlation_heatmap` | Check feature correlation structure |
| pca | `plot_pca_scatter` | Quick embedding overview |
| umap | `plot_umap_scatter` | Nonlinear embedding overview |
| confusion | `plot_confusion_matrix` | Classification performance (counts + normalized) |
| reliability | `plot_reliability_diagram` | Calibration quality |
| workflow | `plot_workflow_dag` | Visualize protocol steps |
| params | `plot_parameter_map` | Inspect flattened config |
| lineage | `plot_data_lineage` | Track input provenance |
| badge | `plot_reproducibility_badge` | Run reproducibility summary |
| batch_drift | `plot_batch_drift` | Batch drift diagnostics |
| stage_diff | `plot_stage_difference_spectra` | Stage-wise spectral differences |
| replicate_similarity | `plot_replicate_similarity` | Replicate consistency |
| temporal_drift | `plot_temporal_drift` | Drift trend over time |
| importance_overlay | `plot_importance_overlay` | Highlight influential bands |
| marker_bands | `plot_marker_bands` | Display selected marker regions |
| coefficient_heatmap | `plot_coefficient_heatmap` | Linear model coefficients |
| feature_stability | `plot_feature_stability` | Stability across folds |
| confidence | `plot_confidence_map` | Prediction confidence map |
| conformal | `plot_conformal_set_sizes` | Conformal set size distribution |
| coverage_efficiency | `plot_coverage_efficiency` | Coverage vs efficiency |
| abstention | `plot_abstention_distribution` | Abstention/reject analysis |

## Output contract

All figures are saved under:

```
{OUTDIR}/figures/<name>.png
{OUTDIR}/figures/<name>.svg
{OUTDIR}/figures/<name>.meta.json
```

Metadata includes a description, input summary, and FoodSpec version.
