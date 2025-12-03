# ML & Chemometrics: Mixture Models and Fingerprinting

This chapter covers compositional and similarity analyses for mixtures, including non-negative least squares (NNLS), MCR-ALS, and fingerprint similarity for library search or QC.

> For notation and symbols used below, see the [Glossary](../glossary.md).

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

## NNLS formulation (what and why)
Given a matrix of reference spectra \(A \in \mathbb{R}^{m\times n}\) (columns = pure components, rows = wavenumbers) and an observed mixture spectrum \(y\in\mathbb{R}^m\), NNLS solves:

\[ \min_{x} \; \|A x - y\|_2^2 \quad \text{subject to}\; x \ge 0. \]

- **Spectroscopy meaning:**
  - **A:** pure/reference spectra (e.g., EVOO, sunflower).
  - **x:** non-negative fractions/coefficients to estimate.
  - **y:** measured mixture spectrum.
- **Why non-negativity?** Concentrations are non-negative; allows a physical interpretation of fractions.
- **Assumptions:** Linear mixture, same preprocessing/cropping for A and y; well-aligned wavenumbers; scatter/scale effects minimized via normalization.

### Minimal code example (NNLS)
```python
import numpy as np
from foodspec.chemometrics.mixture import nnls_mixture

# A: columns are reference spectra (n_points x n_components)
# y: mixture spectrum (n_points,)
coeffs, residual = nnls_mixture(y, A)
print("fractions", coeffs / coeffs.sum(), "residual", residual)
```

### Visualizing NNLS reconstruction
- Plot the observed mixture spectrum **y**, reconstructed spectrum **A @ x̂**, and residual **y - A @ x̂**.
- Good fit: reconstructed overlays y closely; residual shows no systematic structure.
- Report reconstruction error (e.g., RMSE or R²) alongside the plot.

For notation and symbols, see the [Glossary](../glossary.md).

You can create a synthetic demo with two reference spectra, mix them at known fractions, solve NNLS, and overlay observed vs reconstructed. Use either the example oils or synthetic spectra.

## Next steps
- See workflows (mixture analysis, QC) in Part IV for applied recipes.
- Refer back to metrics and validation chapters for interpreting regression performance.
