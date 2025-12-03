# ML & Chemometrics: Mixture Models and Fingerprinting

Compositional analysis decomposes mixtures into fractions of known or unknown components, while fingerprinting compares spectra for QC or search. This page follows the WHAT/WHY/WHEN/WHERE template.

> For notation see the [Glossary](../glossary.md). For plots and metrics see [Metrics & Evaluation](../metrics/metrics_and_evaluation.md) and [Visualization](../visualization/plotting_with_foodspec.md).

## What?
Defines NNLS (non-negative least squares) for single mixtures with known references, MCR-ALS for unsupervised mixtures, and fingerprint similarity (cosine/correlation). Inputs: preprocessed spectra, reference spectra, or libraries. Outputs: fractions/coefficients, reconstructed spectra, similarity scores, and metrics (RMSE/R²).

## Why?
Linear mixtures of food components (oils, adulterants, moisture) can be estimated physically when non-negativity is enforced. Fingerprinting supports QC/search by comparing against libraries.

## When?
**Use:**  
- NNLS: known pure/reference spectra, want non-negative fractions per sample.  
- MCR-ALS: multiple mixtures, components unknown/partially known.  
- Fingerprinting: QC/search against libraries.  
**Limitations:** assumes linear mixing and aligned preprocessing; scatter/scale issues must be minimized; MCR-ALS can be sensitive to initialization.

## Where? (pipeline)
Upstream: consistent preprocessing/cropping/normalization for mixtures and references.  
Model: NNLS or MCR-ALS; fingerprint similarity optional for QC.  
Downstream: reconstruction plots, residual analysis, RMSE/R², stats on ratios/coefficients.  
```mermaid
flowchart LR
  A[Preprocess refs + mixtures] --> B[NNLS / MCR-ALS]
  B --> C[Fractions + reconstruction]
  C --> D[Metrics (RMSE/R²) + plots]
  D --> E[Reporting / stats]
```

## NNLS math & interpretation
Given reference spectra matrix \(A \in \mathbb{R}^{m\times n}\) (columns = pure components, rows = wavenumbers) and mixture \(y \in \mathbb{R}^m\), solve
\[
\min_{x} \|A x - y\|_2^2 \quad \text{s.t. } x \ge 0.
\]
- \(A\): pure/reference spectra (e.g., EVOO, sunflower).  
- \(x\): non-negative fractions/coefficients.  
- \(y\): observed mixture.  
Non-negativity enforces physical interpretability. Assumes linear mixing and matched preprocessing.

### Minimal code example (NNLS)
```python
import numpy as np
from foodspec.chemometrics.mixture import nnls_mixture

coeffs, resid = nnls_mixture(y, A)  # y: (n_points,), A: (n_points, n_components)
fractions = coeffs / coeffs.sum()
```

### Visuals + metrics
- Plot observed mixture **y**, reconstructed **A @ x̂**, and residual **y - A @ x̂**.  
- Good fit: close overlay, residual without structure; quantify with RMSE/R² (see metrics chapter).  
- Reproducible figure: run  
  ```bash
  python docs/examples/visualization/generate_mixture_nnls_figures.py
  ```  
  to save `docs/assets/nnls_overlay.png` and `docs/assets/nnls_residual.png` using synthetic references. Use example oils if desired by swapping in real references.

## MCR-ALS (outline)
- Factorize mixtures matrix \(\mathbf{X} \approx \mathbf{C}\mathbf{S}^\top\) iteratively with non-negativity.  
- Returns concentrations \(\mathbf{C}\) and estimated pure-like spectra \(\mathbf{S}\).  
- Monitor convergence, enforce non-negativity, and compare reconstructed X to data (RMSE, residual structure).

## Fingerprinting
- Cosine/correlation similarities for QC/search against libraries.  
- Plot heatmaps or top-k matches; thresholds should be validated per application.

## Typical plots (with metrics)
- Mixture overlay + residual (report RMSE/R²).  
- Coefficient/fraction bar plots.  
- Similarity heatmaps for fingerprint search.  
- Optional: residual distribution to spot systematic misfit.

## Practical guidance
- Align wavenumbers and preprocessing between mixtures and references.  
- Normalize or scatter-correct before NNLS to reduce scale effects.  
- Start MCR-ALS with sensible initial guesses; check for rotations/scale indeterminacy.  
- Pair visuals with metrics (RMSE/R²) and, if comparing groups, use stats tests on coefficients/ratios (ANOVA/Games–Howell).  
- Document reference provenance; mismatched references yield biased fractions.

## See also
- [Classification & regression](classification_regression.md)  
- [Metrics & evaluation](../metrics/metrics_and_evaluation.md)  
- [Feature extraction](../preprocessing/feature_extraction.md)  
- [Workflow: mixture analysis](../workflows/mixture_analysis.md)  
- [Workflow: calibration/regression](../workflows/calibration_regression_example.md)
