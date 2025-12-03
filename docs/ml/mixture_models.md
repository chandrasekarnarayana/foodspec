# ML & Chemometrics: Mixture Models and Fingerprinting

This chapter covers compositional and similarity analyses for mixtures, including non-negative least squares (NNLS), MCR-ALS, and fingerprint similarity for library search or QC.

## What this chapter covers
- Mixture modeling assumptions (linear mixtures, non-negativity).
- NNLS for single-sample decomposition against known pure spectra.
- MCR-ALS for unsupervised recovery of components and concentrations.
- Fingerprint similarity (cosine/correlation) for mixture screening or library matching.
- Metrics: RMSE/MAE/R², residual analysis, bias.

## Outline
- **Problem framing:** Estimating component fractions from spectra; when reference spectra are known vs unknown.
- **NNLS:** Formulation; use with small pure libraries; sensitivity to alignment/scatter.
- **MCR-ALS:** Alternating updates; initialization, non-negativity clipping; convergence checks.
- **Similarity/fingerprinting:** Cosine/correlation matrices; thresholds for QC or search.
- **Implementation hooks:** `foodspec.chemometrics.mixture`, `foodspec.features.fingerprint`; CLI `mixture` workflow.
- **Reporting:** Predicted vs true plots; residuals; discussing assumptions and limits.

## Next steps
- See workflows (mixture analysis, QC) in Part IV for applied recipes.
- Refer back to metrics and validation chapters for interpreting regression performance.
