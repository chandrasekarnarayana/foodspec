# ML & Chemometrics: PCA and Dimensionality Reduction

Principal Component Analysis (PCA) summarizes high-dimensional spectra into a few orthogonal components. It is a workhorse for exploration, outlier detection, and pre-model checks in food spectroscopy.

## 1. Why dimensionality reduction?
- Spectra can have thousands of wavenumbers; many are correlated.
- PCA finds directions of maximum variance, reducing noise and highlighting structure (clusters by oil type, batch effects, instrument drift).
- Scores plots help visualize separability and spot adulteration or degradation trends before supervised modeling.

## 2. PCA concepts (brief math)
- Center data \(X\) (n_samples × n_features). Covariance \( \Sigma = \frac{1}{n-1} X^\top X \).
- Eigen-decompose \( \Sigma = V \Lambda V^\top \); columns of \(V\) are loadings, \( \Lambda \) are variances.
- Scores \( S = X V \) project spectra; explained variance ratio \( \lambda_i / \sum \lambda \).

## 3. Interpreting scores and loadings
- **Scores plot:** PC1 vs PC2 colored by metadata (oil_type, heating_time). Clusters suggest separability; outliers may be bad spectra or novel samples.
- **Loadings plot:** Loadings vs wavenumber show which bands drive each PC. Peaks in loadings link to chemical groups.
- **Cautions:** PCA is unsupervised; variance may reflect noise or baseline if preprocessing is weak. Ensure baseline/normalization is done first.

## 4. Practical patterns in food spectroscopy
- **Oil authentication:** PC1/PC2 often separate oil families; loadings highlight unsaturation/ester bands.
- **Heating studies:** PC trends can correlate with time/temperature; loadings show oxidation markers.
- **QC/novelty:** Outliers in score space can flag suspect batches or spectral artifacts.

## 5. Example (high level)
```python
from foodspec.chemometrics.pca import run_pca
from foodspec.viz.pca import plot_pca_scores, plot_pca_loadings

pca, res = run_pca(X_proc, n_components=3)
fig_scores = plot_pca_scores(res.scores[:, :2], labels=fs.metadata["oil_type"])
fig_load = plot_pca_loadings(res.loadings[:, 0], wavenumbers=fs.wavenumbers)
```

## 6. Visuals to include
- Scree plot (explained variance).
- Scores plot colored by oil_type or time; loadings vs wavenumber with annotated bands.

## Summary
- PCA reduces dimensionality and reveals structure; interpret scores and loadings in chemical context.
- Good preprocessing (baseline, normalization) is essential to avoid baselines dominating PCs.
- Use PCA for exploration, QC, and as a sanity check before supervised models.

## Further reading
- [Classification & regression](classification_regression.md)
- [Baseline correction](../preprocessing/baseline_correction.md)
- [Normalization & smoothing](../preprocessing/normalization_smoothing.md)
