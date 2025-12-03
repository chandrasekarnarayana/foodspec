# Plotting with FoodSpec

This page summarizes common plot types in FoodSpec and the helper functions to generate them. All helpers return Matplotlib Axes and use simple defaults suitable for scientific reporting.

## Spectra overlays and means
```python
from foodspec.viz import plot_spectra_overlay, plot_mean_with_ci

ax = plot_spectra_overlay(spectra, wavenumbers, labels=sample_ids)
ax = plot_mean_with_ci(spectra, wavenumbers, group_labels=groups, ci=95)
```
Use to inspect raw/preprocessed spectra, baselines, and group differences. Label axes in cm⁻¹ and intensity.

## PCA plots
```python
from foodspec.viz import plot_pca_scores, plot_pca_loadings

ax = plot_pca_scores(scores, labels=classes, components=(1, 2))
ax = plot_pca_loadings(loadings, wavenumbers, components=(1, 2))
```
Use to explore clustering and which bands drive separation.

## Classification diagnostics
```python
from foodspec.viz import plot_confusion_matrix

ax = plot_confusion_matrix(cm, class_labels, normalize=True)
```
Use for classification workflows (oil auth, QC) to see per-class performance.

## Correlation and heatmaps
```python
from foodspec.viz import plot_correlation_heatmap

ax = plot_correlation_heatmap(corr_matrix, labels=feature_names)
```
Use to summarize associations (e.g., ratios vs quality metrics).

## Hyperspectral maps
```python
from foodspec.viz.hyperspectral import plot_hyperspectral_intensity_map

ax = plot_hyperspectral_intensity_map(cube, target_wavenumber=1655, window=5)
```
Use to localize components/defects and visualize spatial heterogeneity.

## Regression calibration
```python
from foodspec.viz import plot_regression_calibration

ax = plot_regression_calibration(y_true, y_pred)
```
Use for calibration/regression tasks to check predicted vs true alignment.

## Where these appear in workflows
- Oil authentication: confusion matrix, PCA scores/loadings, boxplots of ratios.
- Heating: ratio vs time plots (see `viz.heating.plot_ratio_vs_time`), correlation heatmaps.
- Mixtures/calibration: regression calibration plots.
- QC: confusion matrices and spectra overlays for suspect vs reference.
- Hyperspectral: intensity/ratio maps; see [Hyperspectral mapping](../workflows/hyperspectral_mapping.md).

## See also
- [Workflow design](../workflows/workflow_design_and_reporting.md#plots-visualizations)
- [Calibration/regression workflow](../workflows/calibration_regression_example.md)
- [Stats: correlation & mapping](../stats/correlation_and_mapping.md)
