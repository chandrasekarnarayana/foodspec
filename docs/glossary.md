# Glossary of Symbols and Terms

This glossary centralizes notation and acronyms used across FoodSpec. Refer here when reading math-heavy sections (PCA, NNLS, ALS baseline, stats, metrics).

## Symbols
- \(A\): Matrix of reference spectra (columns = pure components) or design matrix.
- \(x\): Coefficient vector (e.g., mixture fractions in NNLS).
- \(y\): Observed spectrum or response vector.
- \(\lambda\): Wavelength (nm).
- \(\tilde{\nu}\): Wavenumber (cm⁻¹).
- \(\Sigma\): Covariance matrix.
- \(V, \Lambda\): Eigenvector and eigenvalue matrices in PCA (\(\Sigma = V \Lambda V^\top\)).
- \(\alpha\): Significance level (e.g., 0.05).
- \(d\): Cohen’s d effect size.
- \(s^2\): Variance.
- \(T\): PCA scores matrix; \(P\): PCA loadings matrix.
- \(f\): F-statistic (ANOVA, embedding between/within ratio analogue).
- \(p_{\text{perm}}\): Permutation p-value.

## Abbreviations & Terms
- **ALS**: Asymmetric Least Squares baseline correction.
- **ANOVA**: Analysis of Variance (group mean comparisons).
- **AUC**: Area Under the Curve (often ROC).
- **CI**: Confidence Interval.
- **DL**: Deep Learning.
- **FDR**: False Discovery Rate.
- **F1**: Harmonic mean of precision and recall.
- **IoU**: Intersection over Union (segmentation accuracy).
- **LOA**: Limits of Agreement (Bland–Altman).
- **MAPE**: Mean Absolute Percentage Error.
- **MAE / RMSE**: Mean Absolute Error / Root Mean Square Error.
- **MCC**: Matthews Correlation Coefficient.
- **MCR-ALS**: Multivariate Curve Resolution – Alternating Least Squares.
- **NNLS**: Non-Negative Least Squares (mixture fractions).
- **OC-SVM**: One-Class Support Vector Machine (novelty detection).
- **PCA**: Principal Component Analysis.
- **PLS / PLS-DA**: Partial Least Squares (Regression / Discriminant Analysis).
- **PR curve**: Precision–Recall curve.
- **QC**: Quality Control.
- **ROC**: Receiver Operating Characteristic.
- **SNR**: Signal-to-Noise Ratio.
- **t-SNE**: t-distributed Stochastic Neighbor Embedding (visualization only).
# Glossary

**Spectrum / Spectra**  
An array of intensities measured across wavenumbers (Raman/FTIR).

**Peak / Band**  
A local maximum or defined region in a spectrum (e.g., 1740 cm⁻¹ carbonyl band).

**Ratio**  
Intensity or area of one peak divided by another (e.g., I\_1742 / I\_2720) to reduce illumination variability.

**RQ (Ratio-Quality) Engine**  
FoodSpec module that computes stability, discriminative power, trends, divergence, minimal panels, and clustering on peaks/ratios.

**CV (Coefficient of Variation)**  
Standard deviation divided by mean (often expressed as %). Used for stability/reproducibility.

**MAD (Median Absolute Deviation)**  
Robust dispersion measure, less sensitive to outliers than standard deviation.

**FDR (False Discovery Rate)**  
Multiple-testing correction controlling expected proportion of false positives across many p-values.

**Effect size**  
Quantifies the magnitude of a difference (e.g., Cohen’s d, slope delta), complementing p-values.

**Batch / Group**  
Set of samples sharing an instrument/run/lot; kept intact in batch-aware validation to avoid leakage.

**Harmonization**  
Aligning spectra from different instruments/runs (wavenumber calibration, power normalization) to make them comparable.

**HSI (Hyperspectral Imaging)**  
3D data (x, y, wavenumber) capturing spatially resolved spectra; often segmented into regions of interest (ROIs).

**Frozen model**  
Serialized model package containing preprocessing, feature definitions, weights, and metadata for prediction reuse.

**Protocol**  
YAML/JSON recipe defining steps (preprocess, harmonize, QC, HSI, RQ, output), expected columns, and validation strategy.

**Bundle**  
Run folder with report(s), figures, tables, metadata.json, index.json, logs, and models (if trained).
