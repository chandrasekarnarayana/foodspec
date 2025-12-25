# Workflow: Mixture Analysis

> New to workflow design? See [Designing & reporting workflows](workflow_design_and_reporting.md).
> For model/evaluation guidance, see [ML & DL models](../ml/models_and_best_practices.md) and [Metrics & evaluation](../metrics/metrics_and_evaluation.md).

Mixture analysis estimates component fractions (e.g., EVOO–sunflower blends) from spectra. This workflow uses NNLS when pure references exist and MCR-ALS when they do not.

Suggested visuals: predicted vs true scatter, residual plots, correlation heatmaps for predicted/true fractions. See [Plots guidance](workflow_design_and_reporting.md#plots-visualizations).
For troubleshooting (peak alignment, imbalance of mixtures), see [Common problems & solutions](../troubleshooting/common_problems_and_solutions.md).

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
- **Quantitative:** Report reconstruction RMSE/R²; residual norm from NNLS; optional permutation p_perm on between/within separation if visualizing embeddings of fractions/components. Link to [Metrics](../metrics/metrics_and_evaluation.md) and [Stats](../stats/overview.md) for regression diagnostics.
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
- [Normalization & smoothing](../preprocessing/normalization_smoothing.md)
- [Mixture models & fingerprinting](../ml/mixture_models.md)
- [Model evaluation](../ml/model_evaluation_and_validation.md)
