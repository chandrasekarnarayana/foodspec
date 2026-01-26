# Multivariate Reporting

FoodSpec automatically renders multivariate outputs when a run directory contains the standard artifacts under `multivariate/` and `qc/`.

## Expected Artifacts
- `multivariate/<method>/scores.csv`
- `multivariate/<method>/loadings.csv`
- `multivariate/<method>/summary.json`
- `qc/multivariate_outliers.csv` (optional)
- `qc/multivariate_drift.csv` (optional)

## Generated Figures
- `figures/multivariate/pca_scores_label.png`
- `figures/multivariate/pca_scores_batch.png`
- `figures/multivariate/pca_loadings_pc1.png`
- Heatmaps for multi-component loadings when available.

## YAML Snippet
```yaml
qc:
  multivariate:
    enabled: true
    outliers:
      method: hotelling_t2
      alpha: 0.01
      policy: flag
    drift:
      enabled: true
      metric: centroid_l2
      warn_threshold: 2.0
      fail_threshold: 4.0
```

If these files exist, the HTML report and experiment card will include multivariate summaries, QC status, and key risks automatically.
