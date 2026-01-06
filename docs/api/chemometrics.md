# Chemometrics API

Multivariate analysis tools: PCA, PLS, and mixture modeling.

The `foodspec.chemometrics` module provides classical chemometric methods for dimensionality reduction, classification, and quantitative analysis.

## Principal Component Analysis (PCA)

### run_pca

Perform PCA on spectral data with comprehensive outputs.

::: foodspec.chemometrics.pca.run_pca
    options:
      show_source: false
      heading_level: 4

## Partial Least Squares (PLS)

### make_pls_da

Create PLS Discriminant Analysis classifier.

::: foodspec.chemometrics.models.make_pls_da
    options:
      show_source: false
      heading_level: 4

### make_pls_regression

Create PLS regression model.

::: foodspec.chemometrics.models.make_pls_regression
    options:
      show_source: false
      heading_level: 4

## Mixture Modeling

### mcr_als

Multivariate Curve Resolution - Alternating Least Squares.

::: foodspec.chemometrics.mixture.mcr_als
    options:
      show_source: false
      heading_level: 4

### nnls_mixture

Non-negative least squares mixture deconvolution.

::: foodspec.chemometrics.mixture.nnls_mixture
    options:
      show_source: false
      heading_level: 4

## Variable Importance

### calculate_vip

Variable Importance in Projection scores for PLS models.

::: foodspec.chemometrics.vip.calculate_vip
    options:
      show_source: false
      heading_level: 4

## See Also

- **[Chemometrics Methods](../methods/chemometrics/pca_and_dimensionality_reduction.md)** - Detailed methodology
- **[PCA Guide](../methods/chemometrics/pca_and_dimensionality_reduction.md)** - PCA concepts
- **[Examples](../examples_gallery.md)** - Chemometric workflows
