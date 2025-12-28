# Plotting with FoodSpec

This page summarizes common plot types in FoodSpec and the helper functions to generate them. All helpers return Matplotlib Axes and use simple defaults suitable for scientific reporting.

> For notation see the [Glossary](../reference/glossary.md). Pair every plot with the quantitative metric it supports (see [Metrics & Evaluation](../reference/metrics_reference.md)).

## What?
Catalog of visualization helpers for spectra, embeddings, classification/regression diagnostics, ratios, and hyperspectral maps.

## Why?
Spectral interpretation benefits from paired visuals and metrics: overlays to check preprocessing, PCA/t-SNE to view structure (with silhouette/between-within metrics), classification/regression plots to diagnose errors and agreement, and ratio plots to show chemistry-driven separations.

## When?
Use after preprocessing/features are fixed, and when summarizing CV/test results or exploratory structure. Limitations: visuals are qualitative unless paired with metrics; avoid over-interpreting small-n plots—add CIs/bootstraps where possible.

## Spectra overlays and means
```python
from foodspec.viz import plot_spectra_overlay, plot_mean_with_ci

ax = plot_spectra_overlay(spectra, wavenumbers, labels=sample_ids)
ax = plot_mean_with_ci(spectra, wavenumbers, group_labels=groups, ci=95)
```
Use to inspect raw/preprocessed spectra, baselines, and group differences. Axes: cm⁻¹ and intensity.

## PCA and embedding plots
```python
from foodspec.viz import plot_pca_scores, plot_pca_loadings

ax = plot_pca_scores(scores, labels=classes, components=(1, 2))
ax = plot_pca_loadings(loadings, wavenumbers, components=(1, 2))
```
Interpretation: scores = sample structure; loadings = bands driving separation. Pair with silhouette/between-within F-like stats and p_perm (metrics chapter). For t-SNE, use the same reading: clusters need metric support.

## Classification diagnostics
```python
from foodspec.viz import plot_confusion_matrix, plot_roc_curve

ax = plot_confusion_matrix(cm, class_labels, normalize=True)
ax = plot_roc_curve(fpr, tpr, auc_value)
```
Confusion matrix shows per-class errors; ROC/PR curves summarize ranking for balanced/imbalanced tasks.

## Correlation and heatmaps
```python
from foodspec.viz import plot_correlation_heatmap

ax = plot_correlation_heatmap(corr_matrix, labels=feature_names)
```
Use to summarize associations (ratios vs quality metrics, PCs vs lab values).

## Ratio plots
```python
from foodspec.viz import plot_ratio_by_group, plot_ratio_scatter, plot_ratio_vs_continuous

ax1 = plot_ratio_by_group(ratios, group_labels, kind="box")
ax2 = plot_ratio_scatter(ratio_x, ratio_y, group_labels)
ax3 = plot_ratio_vs_continuous(ratios, cont_var)
```
Interpretation: box/violin separation suggests group differences (support with ANOVA/Games–Howell + effect sizes). Ratio–ratio scatter reveals clusters/regimes; ratio vs continuous shows trends (support with regression metrics and CI on slope).

## Hyperspectral maps
```python
from foodspec.viz.hyperspectral import plot_hyperspectral_intensity_map

ax = plot_hyperspectral_intensity_map(cube, target_wavenumber=1655, window=5)
```
Use to localize components/defects and visualize spatial heterogeneity; pair with summary stats per region if available.

## Regression calibration and agreement
```python
from foodspec.viz import (
    plot_regression_calibration,
    plot_calibration_with_ci,
    plot_bland_altman,
)

ax = plot_regression_calibration(y_true, y_pred)
ax = plot_calibration_with_ci(y_true, y_pred)
ax = plot_bland_altman(y_true, y_pred)
```
Good fit hugs the 1:1 line; CI bands convey uncertainty; Bland–Altman bias near 0 with tight limits indicates agreement. Trends in diff vs mean suggest scale/offset issues.

## Reproducible figure generation
- Embeddings: `python docs/examples/visualization/generate_embedding_figures.py` → PCA scores/loadings and t-SNE with accompanying metrics.  
- Mixtures/NNLS: (describe or run) a synthetic or example-oils script to save mixture overlay + residuals.  
- Ratios: reuse example oils to create ratio-by-group and ratio-vs-continuous plots.  
- Classification/regression: run simple PCA+SVM or PLS calibration examples and call the viz helpers above.

## Where these appear in workflows
- Oil authentication: confusion matrix, PCA scores/loadings, boxplots of ratios.  
- Heating: ratio vs time plots (`viz.heating.plot_ratio_vs_time`), correlation heatmaps.  
- Mixtures/calibration: regression calibration plots with CI; residual overlays.  
- QC: confusion matrices and spectra overlays for suspect vs reference.  
- Hyperspectral: intensity/ratio maps; see [Hyperspectral mapping](../workflows/spatial/hyperspectral_mapping.md).

## See also
- [Workflow design](../workflows/workflow_design_and_reporting.md#plots-visualizations)  
- [Calibration/regression workflow](../workflows/quantification/calibration_regression_example.md)  
- [Stats: correlation & mapping](../stats/correlation_and_mapping.md)  
- [Metrics & evaluation](../reference/metrics_reference.md)
