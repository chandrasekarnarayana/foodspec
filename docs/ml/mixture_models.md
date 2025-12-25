# ML & Chemometrics: Mixture Models and Fingerprinting

Compositional analysis decomposes mixtures into fractions of known or unknown components, while fingerprinting compares spectra for QC or search. This page follows the WHAT/WHY/WHEN/WHERE template.

> For notation see the [Glossary](../09-reference/glossary.md). For plots and metrics see [Metrics & Evaluation](../metrics/metrics_and_evaluation/) and [Visualization](../visualization/plotting_with_foodspec.md).

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

---

## When Results Cannot Be Trusted

⚠️ **Red flags for mixture decomposition validity:**

1. **Reference spectra mismatched to real mixtures (using pure oil standards to decompose adulterated oils with unexpected components)**
   - NNLS/MCR-ALS estimates are only valid if reference set spans true composition space
   - Missing components force solution to fit residuals with existing references, producing wrong fractions
   - **Fix:** Ensure reference library includes all expected components; validate against known mixtures

2. **Degenerate solutions (fractions sum to 1.0 with near-zero negative values, suggesting numerical instability)**
   - Ill-conditioning (nearly collinear references) allows multiple valid solutions
   - Small perturbations in data yield different solutions
   - **Fix:** Check condition number of reference matrix; validate solution stability with bootstrap or cross-validation

3. **Estimated fractions outside [0, 1] (negative fraction or >100% total) reported without warning**
   - NNLS constrains to non-negative, but if total doesn't equal 1.0, assumes unaccounted component
   - Unconstrained solving (OLS) may produce negative fractions, indicating poor fit or leakage
   - **Fix:** Use constrained solver (NNLS); check sum-to-1 constraint; validate on known mixtures

4. **Rotational ambiguity in MCR-ALS unresolved (multiple equivalent solutions with same fit, different spectra)**
   - MCR can have rank deficiency → multiple (A, S) pairs fit data equally well
   - Reported spectra may not be true pure component spectra
   - **Fix:** Test for rotational ambiguity; use constraints (non-negativity, bounds) to enforce unique solution; validate spectra chemically

5. **Preprocessing choices not disclosed or not matched to references**
   - If samples preprocessed differently from references, NNLS produces biased fractions
   - Example: baseline correction on samples but not references → fractions shift
   - **Fix:** Preprocess samples and references identically; freeze preprocessing before decomposition

6. **No validation on known mixtures (deploying model without testing on mixtures of known composition)**
   - NNLS/MCR can produce chemically plausible fractions even on synthetic data
   - Only validation on truly known mixtures confirms model works
   - **Fix:** Test on lab-prepared mixtures of known composition; report agreement (RMSE, R²) against true fractions

7. **Small number of references (2 pure components) used to estimate many mixture fractions**
   - With only 2 references and many wavenumbers, system is typically underdetermined; many solutions fit
   - Tiny errors in spectra or preprocessing cause large fraction changes
   - **Fix:** Increase reference library; use more wavenumber regions; apply regularization

8. **Visualization doesn't match quantitative metrics (visual inspection suggests good fit, but RMSE is high)**
   - Residual plots can be misleading; always pair with numeric metrics
   - High RMSE despite visually acceptable fit may indicate systematic bias (e.g., baseline residual)
   - **Fix:** Report both visual residuals and RMSE/R²; visualize all mixtures, not just representative ones
- Document reference provenance; mismatched references yield biased fractions.

## See also
- [Classification & regression](classification_regression.md)  
- [Metrics & evaluation](../metrics/metrics_and_evaluation/)  
- [Feature extraction](../preprocessing/feature_extraction.md)  
- [Workflow: mixture analysis](../workflows/mixture_analysis.md)  
- [Workflow: calibration/regression](../workflows/calibration_regression_example.md)
