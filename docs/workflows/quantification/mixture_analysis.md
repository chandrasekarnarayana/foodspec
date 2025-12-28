# Workflow: Mixture Analysis

## 📋 Standard Header

**Purpose:** Estimate component fractions in mixtures (e.g., olive oil adulterated with sunflower oil) using spectral unmixing.

**When to Use:**
- Quantify adulteration level in food products
- Determine blending ratios in multi-component samples
- Monitor process streams for composition drift
- Validate supplier blend specifications
- Estimate impurity levels when pure references available

**Inputs:**
- Format: HDF5 or CSV with mixture spectra + (optional) pure component spectra
- Required metadata: `mixture_id`; optional: true fractions (for validation)
- Pure spectra: One spectrum per component (if using NNLS)
- Wavenumber range: Must align across mixtures and pure spectra (typically 600–1800 cm⁻¹)
- Min samples: 10+ mixtures; 1 pure spectrum per component (if using NNLS)

**Outputs:**
- fractions.csv — Estimated component fractions for each mixture
- pred_vs_true.png — Scatter plot (if ground truth available)
- residual_plot.png — Reconstruction error per mixture
- spectral_reconstruction.png — Observed vs reconstructed spectra
- report.md — RMSE, R², correlation for each component

**Assumptions:**
- Spectra are linear combinations of pure components (Beer-Lambert holds)
- Pure spectra representative of components in mixtures (no matrix effects)
- Wavenumbers aligned (no shifts between mixtures and pure spectra)
- No interactions between components (no chemical reactions in mixtures)

---

## 🔬 Minimal Reproducible Example (MRE)

### Option A: NNLS with Pure References

```python
import numpy as np
import matplotlib.pyplot as plt
from foodspec.chemometrics.mixture import run_mixture_analysis_workflow
from foodspec.viz.mixture import plot_pred_vs_true
from examples.mixture_analysis_quickstart import _synthetic_mixtures

# Generate synthetic mixtures with known fractions
mixtures, pure_spectra, true_coeffs = _synthetic_mixtures(n_mixtures=30)
print(f"Mixtures: {mixtures.x.shape[0]} spectra")
print(f"Pure components: {pure_spectra.x.shape[0]}")
print(f"True fractions shape: {true_coeffs.shape}")

# Run NNLS workflow
result = run_mixture_analysis_workflow(
    mixtures=mixtures.x,
    pure_spectra=pure_spectra.x,
    mode="nnls"  # Non-negative least squares
)

# Extract results
pred_fractions = result["coefficients"]
residual_norms = result["residual_norms"]

# Evaluate if ground truth available
from sklearn.metrics import r2_score, mean_squared_error
for i, comp in enumerate(['Component A', 'Component B']):
    r2 = r2_score(true_coeffs[:, i], pred_fractions[:, i])
    rmse = np.sqrt(mean_squared_error(true_coeffs[:, i], pred_fractions[:, i]))
    print(f"{comp}: R²={r2:.3f}, RMSE={rmse:.3f}")

# Plot predicted vs true
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (ax, comp) in enumerate(zip(axes, ['Component A', 'Component B'])):
    plot_pred_vs_true(
        true_coeffs[:, i],
        pred_fractions[:, i],
        ax=ax
    )
    ax.set_title(comp)
    ax.set_xlabel("True Fraction")
    ax.set_ylabel("Predicted Fraction")
plt.tight_layout()
plt.savefig("mixture_pred_vs_true.png", dpi=150, bbox_inches='tight')
print("Saved: mixture_pred_vs_true.png")
```

**Expected Output:**
```yaml
Mixtures: 30 spectra
Pure components: 2
True fractions shape: (30, 2)

Component A: R²=0.952, RMSE=0.042
Component B: R²=0.948, RMSE=0.045

Saved: mixture_pred_vs_true.png
```

### Option B: MCR-ALS without Pure References

```python
from foodspec.chemometrics.mcr import run_mcr_als

# Same mixture data, but don't use pure spectra
mixtures, _, true_coeffs = _synthetic_mixtures(n_mixtures=30)

# Run MCR-ALS (alternating least squares)
result_mcr = run_mcr_als(
    mixtures.x,
    n_components=2,  # Must specify number of components
    max_iter=100,
    tol=1e-4
)

pred_fractions_mcr = result_mcr["coefficients"]
recovered_spectra = result_mcr["components"]

print(f"MCR-ALS converged in {result_mcr['n_iter']} iterations")
print(f"Final residual: {result_mcr['residual']:.4f}")

# Note: MCR-ALS may recover components in different order or with scaling
# Alignment to true fractions requires post-processing
```

---

## ✅ Validation & Sanity Checks

### Success Indicators

**Predicted vs True (if ground truth available):**
- ✅ R² > 0.90 for each component
- ✅ RMSE < 0.05 (5% error in fraction estimation)
- ✅ Points cluster tightly around diagonal (y = x line)

**Residual Analysis:**
- ✅ Residual norms small and uniform across mixtures
- ✅ No systematic patterns in residual spectra (random noise only)
- ✅ Reconstructed spectra visually match observed spectra

**Fractions Sum to 1:**
- ✅ Sum of predicted fractions ≈ 1.0 for each mixture (within 0.05)
- ✅ All fractions ≥ 0 (non-negativity constraint satisfied)
- ✅ No fractions > 1.0 (physically impossible)

### Failure Indicators

**⚠️ Warning Signs:**

1. **R² < 0.70 or RMSE > 0.15**
   - Problem: Poor prediction; model not capturing mixture composition
   - Fix: Check wavenumber alignment; verify pure spectra quality; increase spectral resolution

2. **Predicted fractions sum >> 1.0 or << 1.0**
   - Problem: Scaling issue; normalization mismatch; missing component
   - Fix: Renormalize pure spectra; check if mixture contains unlabeled component

3. **Negative fractions (NNLS should prevent this)**
   - Problem: Algorithm not converging; numerical instability
   - Fix: Increase iteration limit; check for collinear pure spectra; try regularization

4. **High residuals for specific mixtures (outliers)**
   - Problem: Mixture contains impurity; pure spectra not representative; spectral shift
   - Fix: Remove outlier mixtures; check for temperature/pH effects; verify sample quality

5. **MCR-ALS components don't match pure spectra**
   - Problem: Component order ambiguous; scaling arbitrary; local minimum
   - Fix: Use better initialization; increase iterations; compare spectral features manually

### Quality Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|--------|
| R² (per component) | 0.80 | 0.92 | 0.98 |
| RMSE (fraction) | < 0.10 | < 0.05 | < 0.02 |
| Residual Norm | < 5% | < 2% | < 0.5% |
| Fraction Sum Deviation | < 0.10 | < 0.05 | < 0.02 |

---

## ⚙️ Parameters You Must Justify

### Critical Parameters

**1. Unmixing Method**
- **Parameter:** `mode` ("nnls" or "mcr_als")
- **Default:** "nnls" if pure spectra available
- **When to adjust:** Use MCR-ALS if pure spectra unknown or unavailable
- **Justification:** "Non-negative least squares (NNLS) was used to estimate component fractions, as pure spectra were available and Beer-Lambert linearity assumed."

**2. Number of Components (MCR-ALS only)**
- **Parameter:** `n_components`
- **No default:** Must specify
- **When to adjust:** Based on prior knowledge or exploratory PCA (scree plot)
- **Justification:** "Two components were specified based on known binary mixture composition (olive + sunflower oils)."

**3. Spectral Alignment**
- **Parameter:** Wavenumber range, interpolation
- **Critical:** Must align mixtures and pure spectra
- **Justification:** "All spectra were interpolated to a common wavenumber grid (600–1800 cm⁻¹, 1 cm⁻¹ resolution) to ensure Beer-Lambert additivity."

**4. Normalization**
- **Parameter:** Method (area, max, L2) applied to pure spectra
- **Default:** Area normalization (unit area under curve)
- **When to adjust:** Use L2 if intensity scaling consistent; use max if peak heights comparable
- **Justification:** "Pure spectra were area-normalized to unit integral, ensuring fractions represent volume/mass ratios."

**5. Convergence Criteria (MCR-ALS)**
- **Parameter:** `tol` (tolerance), `max_iter`
- **Default:** tol=1e-4, max_iter=100
- **When to adjust:** Increase max_iter if not converging; tighten tol if precision needed
- **Justification:** "MCR-ALS was terminated when relative change in residual < 1e-4 or 100 iterations reached."

---

```mermaid
flowchart LR
  subgraph Data
    A[Raw mixtures] --> A2[Optional pure refs]
  end
  subgraph Preprocess
    B[Baseline + smoothing + norm + align]
  end
  subgraph Features
    C[Stay in spectra or optional PCA/ratios]
  end
  subgraph Model/Stats
    D[NNLS (with refs) or MCR-ALS]
    E[Metrics: RMSE, R²; residuals]
  end
  subgraph Report
    F[Pred vs true + residual overlays + report.md]
  end
  A --> B --> C --> D --> E --> F
  A2 --> D
```

## 1. Problem and dataset
- **Why labs care:** Quantify adulteration level; determine blending ratios; monitor process streams.
- **Inputs:** Mixture spectra; optional pure/reference spectra. Ground truth fractions if available for evaluation.
- **Typical size:** Dozens of mixtures; references for each component if using NNLS.

## 2. Pipeline (default)
- **Preprocessing:** Baseline → smoothing → normalization; ensure wavenumbers align across mixtures/pure spectra.
- **Methods:**
  - **NNLS:** If pure spectra known. Solve \( \min \| x - S c \|_2 \) s.t. \( c \ge 0 \).
  - **MCR-ALS:** If pure spectra unknown; alternating least squares with non-negativity; requires n_components.
- **Outputs:** Coefficients/fractions, residual norms, relative reconstruction error.

## 3. Python example (synthetic)
```python
from foodspec.chemometrics.mixture import run_mixture_analysis_workflow
from examples.mixture_analysis_quickstart import _synthetic_mixtures

mix, pure, true_coeffs = _synthetic_mixtures()
res = run_mixture_analysis_workflow(mixtures=mix.x, pure_spectra=pure.x, mode="nnls")
print("Coefficients:\\n", res["coefficients"])
print("Residual norms:", res["residual_norms"])
```

## 4. CLI example (with config)
Create `examples/configs/mixture_quickstart.yml`:
```yaml
mixture_hdf5: libraries/mixtures.h5
pure_hdf5: libraries/pure_refs.h5
mode: nnls
```
Run:
```bash
foodspec mixture --config examples/configs/mixture_quickstart.yml --output-dir runs/mixture_demo
```
Outputs: coefficients CSV, residuals, optional reconstruction plots.

## 5. Interpretation
- Compare predicted vs true fractions (if known); report RMSE/MAE and R².
- Inspect residuals; large or structured residuals may indicate missing components or misalignment.
- Main figure: predicted vs true scatter; Supplement: residual plots, spectra overlays.

### Qualitative & quantitative interpretation
- **Qualitative:** Overlay observed mixture vs NNLS reconstruction; residual should look like noise, not structured bands. Predicted vs true fractions plot should follow 1:1 line.
- **Quantitative:** Report reconstruction RMSE/R²; residual norm from NNLS; optional permutation p_perm on between/within separation if visualizing embeddings of fractions/components. Link to [Metrics](../../reference/metrics_reference.md) and [Stats](../../methods/statistics/overview.md) for regression diagnostics.
- **Reviewer phrasing:** “NNLS reconstruction overlays the observed spectrum with RMSE = …; predicted fractions track true values (R² = …); residuals show no systematic misfit.”

## Summary
- Use NNLS when pure references are available; otherwise MCR-ALS with chosen n_components.
- Align wavenumbers and preprocess consistently before solving.
- Report fractions, residuals, and assumptions clearly.

## Statistical analysis
- **Why:** Assess how well predicted fractions align with truth; test differences across mixtures if grouped.
- **Example (correlation/regression on predicted vs true):**
```python
from foodspec.chemometrics.mixture import run_mixture_analysis_workflow
from examples.mixture_analysis_quickstart import _synthetic_mixtures
import numpy as np
import pandas as pd
from foodspec.stats import compute_correlations

mix, pure, true_coeffs = _synthetic_mixtures()
res = run_mixture_analysis_workflow(mixtures=mix.x, pure_spectra=pure.x, mode="nnls")
pred = res["coefficients"][:, 0]  # fraction of first component
df = pd.DataFrame({"pred": pred, "true": true_coeffs[:, 0]})
corr = compute_correlations(df, ("pred", "true"), method="pearson")
print(corr)
```
- **Interpretation:** High correlation (and low residual norms) indicates accurate mixture estimation. Report RMSE/MAE and consider ANOVA if comparing multiple pipelines.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for mixture analysis workflow:**

1. **Pure component references not validated (assuming reference spectra are true pure components)**
   - Impure references (contamination, oxidation, partial adulteration) bias fraction estimates
   - NNLS/MCR produces wrong fractions if references don't span true composition
   - **Fix:** Validate references independently (HPLC, mass spec); document purity

2. **Mixture fractions estimated without checking constraints (fractions sum to >100% or include negative values)**
   - Unconstrained solving produces chemically implausible solutions
   - Indicates ill-conditioning, missing components, or data errors
   - **Fix:** Use constrained NNLS; enforce sum-to-1 constraint; investigate violations

3. **Model tested only on synthetic mixtures or known reference blends (not real food samples)**
   - Real samples have components not in reference library
   - Forcing solution into limited reference set produces biased fractions
   - **Fix:** Test on real food samples; compare estimates to orthogonal method (HPLC, GC-MS); report agreement

4. **Number of components not determined independently (assuming n_components = n_references without validation)**
   - Rank deficiency can hide true component number
   - Assuming fixed components may miss unexpected contaminants
   - **Fix:** Estimate rank from data (SVD, scree plot); validate with independent chemical analysis

5. **Preprocessing applied to mixture differently than references (sample has baseline correction, references don't)**
   - Spectral mismatch produces biased fractions
   - NNLS cannot compensate for preprocessing inconsistency
   - **Fix:** Preprocess samples and references identically; freeze parameters before mixture analysis

6. **Mixture proportions highly variable across replicates without investigation**
   - High fold-to-fold variability (e.g., 30% ± 20%) indicates method instability
   - May reflect preprocessing sensitivity or noise
   - **Fix:** Investigate variability sources; test preprocessing robustness; increase replication/averaging

7. **No detection limit study (claiming method can measure 1% admixture without validation)**
   - Method uncertainty depends on noise, reference quality, and mixture complexity
   - Claimed precision may exceed actual detectability
   - **Fix:** Test on mixtures near suspected limit; report limit of detection/quantitation; validate with orthogonal methods

8. **Residuals not visualized or checked (only RMSE reported, no residual spectrum inspection)**
   - Systematic residuals indicate model failure or missing components
   - Residual spectra reveal what the mixture model can't explain
   - **Fix:** Plot residuals vs wavenumber; inspect for systematic patterns; investigate large residuals

## Further reading
- [Normalization & smoothing](../../methods/preprocessing/normalization_smoothing.md)
- [Mixture models & fingerprinting](../../methods/chemometrics/mixture_models.md)
- [Model evaluation](../../methods/chemometrics/model_evaluation_and_validation.md)
