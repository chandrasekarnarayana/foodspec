# ML & Chemometrics: PCA and Dimensionality Reduction

> For notation and symbols used below, see the [Glossary](../../reference/glossary.md).

## What?
PCA projects high-dimensional spectra/features into orthogonal components capturing variance; outputs scores (samples in PC space), loadings (feature contributions), explained variance, and visualizations (scores/loadings, scree, optional t-SNE). Used for exploration, denoising, and as input to downstream ML.

## Why?
Spectra are high-dimensional and correlated; PCA reveals structure (clusters/outliers), highlights important bands, and checks preprocessing quality. Visuals are qualitative; pair with quantitative metrics (silhouette, between/within ratio/F-like statistic, p_perm).

## When?
**Use when** exploring class structure, reducing dimensionality before ML, or interpreting band contributions.  
**Limitations:** linear method; sensitive to scaling/baseline; t-SNE is visualization-only and parameter-sensitive—pair with metrics.

## Where? (pipeline)
Upstream: preprocessing (baseline/norm), feature extraction (peaks/ratios).  
Downstream: scores/loadings plots, silhouette/between-within metrics, ML models.
```mermaid
flowchart LR
  A[Features (spectra/ratios)] --> B[PCA / t-SNE (optional)]
  B --> C[Scores + loadings + metrics]
  C --> D[Visualization + ML]
```

## PCA concepts (brief math)
- Center data \(X\) (n_samples × n_features). Covariance \( \Sigma = \frac{1}{n-1} X^\top X \).
- Eigen-decompose \( \Sigma = V \Lambda V^\top \); columns of \(V\) are loadings, \( \Lambda \) are variances.
- Scores \( S = X V \); explained variance ratio \( \lambda_i / \sum \lambda \).
- PLS (for calibration) maximizes covariance between \(X\) and \(Y\); see regression docs.

## Interpreting scores and loadings
- **Scores plot:** PC1 vs PC2 colored by metadata. Clusters suggest separability; outliers may be bad spectra or novel samples.
- **Loadings plot:** Loadings vs wavenumber show bands driving each PC; relate peaks to vibrational modes ([Spectroscopy basics](../../theory/spectroscopy_basics.md)).
- **Worked example:** If oil A and B separate along PC1 and PC1 loadings peak at ~1655 cm⁻¹ (C=C stretch), that band contributes to A vs B separation.

## Practical patterns
- **Oil authentication:** PC1/PC2 often separate oil families; loadings highlight unsaturation/ester bands.
- **Heating:** PC trends correlate with time/temperature; loadings show oxidation markers.
- **QC/novelty:** Outliers in score space flag suspect batches or artifacts.

## Example (PCA + metrics)
```python
from foodspec.chemometrics.pca import run_pca
from foodspec.viz.pca import plot_pca_scores, plot_pca_loadings
from foodspec.metrics import (
    compute_embedding_silhouette,
    compute_between_within_ratio,
    compute_between_within_stats,
)

pca, res = run_pca(X_proc, n_components=3)
plot_pca_scores(res.scores[:, :2], labels=fs.metadata["oil_type"])
plot_pca_loadings(res.loadings[:, 0], wavenumbers=fs.wavenumbers)
sil = compute_embedding_silhouette(res.scores[:, :2], fs.metadata["oil_type"])
bw = compute_between_within_ratio(res.scores[:, :2], fs.metadata["oil_type"])
stats = compute_between_within_stats(res.scores[:, :2], fs.metadata["oil_type"], n_permutations=200)
print(sil, bw, stats["f_stat"], stats["p_perm"])
```

## Visuals
- **Scree plot:** Explained variance vs component index (`res.explained_variance_ratio_`).
- **Scores plot:** PC1 vs PC2 colored by metadata; read clustering/overlap; pair with silhouette/between-within stats.  
   ![PCA scores](../../assets/pca_scores.png)
- **Loadings plot:** Loadings vs wavenumber; peaks indicate bands driving separation.  
   ![PCA loadings](../../assets/pca_loadings.png)
- **Optional t-SNE:** Visual-only; always pair with metrics (silhouette, between/within, p_perm).  
   ![t-SNE embedding](../../assets/tsne_scores.png)

## Reproducible figures
- Run `python docs/examples/visualization/generate_embedding_figures.py` to regenerate synthetic PCA/t-SNE figures (`pca_scores.png`, `pca_loadings.png`, `tsne_scores.png`).
- For real data (e.g., oils), run PCA after preprocessing; color by oil_type/time; compute silhouette/between-within stats alongside plots.

## Summary
- PCA reduces dimensionality and reveals structure; interpret scores/loadings in chemical context.
- Good preprocessing is essential; variance may otherwise reflect baseline/noise.
- Use PCA/t-SNE for exploration/QC; pair every plot with quantitative metrics (silhouette, between/within ratio/F-stat, p_perm) for defensible interpretation.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for PCA and dimensionality reduction:**

1. **PCA applied to unscaled features (large-magnitude bands dominate, small bands hidden)**
   - Unscaled PCA gives high variance features (e.g., strong C=O band) top components
   - Small but informative features (e.g., weak overtones) contribute little
   - **Fix:** Always standardize (unit variance) or normalize before PCA; document scaling choice

2. **Batch effects not removed before PCA (batch drift appears as PC1, obscuring biology)**
   - Systematic batch variation (instrument age, temperature) can dominate PCA
   - Biological signal hidden in lower components
   - **Fix:** Apply batch correction (ComBat, SVA) before PCA, or visualize/color by batch to interpret batch effects

3. **Number of components chosen by eye ("PC1 + PC2 look separated")**
   - Subjective choice risks overfitting; overstating signal clarity
   - Objective criteria (cumulative variance, scree plot elbow, cross-validation) more defensible
   - **Fix:** Use elbow method, silhouette score, or cross-validation to choose n_components

4. **Outliers not investigated (one sample far from others in PC space)**
   - Outliers can be real (damaged sample, contamination) or artifacts (processing error)
   - Outliers dominate PC1, compress other samples
   - **Fix:** Visualize outliers; check for data errors; consider robust PCA or outlier removal

5. **Loadings interpretation without domain knowledge (PC1 loading high at random peaks)**
   - Loadings can be noisy; high loadings at weak/noisy regions don't indicate importance
   - True informative bands should align with domain knowledge (expected biochemical changes)
   - **Fix:** Cross-check loadings with domain expertise; only trust loadings for strong, consistent peaks

6. **t-SNE/UMAP used without understanding non-linearity can create artifacts**
   - t-SNE/UMAP can exaggerate cluster separation; apparent clusters may dissolve under perturbation
   - Metrics (silhouette, Davies–Bouldin) more objective than visual inspection
   - **Fix:** Use t-SNE/UMAP for exploration only; validate clusters with stability analysis and metrics

7. **Inference on low-dimensional PCA projections (e.g., running statistics on PC1 without considering full space)**
   - Information loss in dimensionality reduction; statistical tests on PCs don't reflect full data
   - Example: group difference in PC1 doesn't mean groups differ significantly in high-D space
   - **Fix:** Run inference on original data or account for dimensionality reduction in models

8. **No explanation of variance explained by noise or batch ("PC1 explains 60%, but this is just drift")**
   - High cumulative variance (>90% in 3 PCs) doesn't guarantee signal quality
   - Can reflect preprocessing artifacts (e.g., baseline residuals) or batch effects
   - **Fix:** Visualize by batch/time; apply batch correction; compute signal-to-noise on residuals

## Further reading
- [Classification & regression](classification_regression.md)
- [Baseline correction](../preprocessing/baseline_correction.md)
- [Normalization & smoothing](../preprocessing/normalization_smoothing.md)
