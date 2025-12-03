# Workflow: Mixture Analysis

> New to workflow design? See [Designing & reporting workflows](workflow_design_and_reporting.md).
> For model/evaluation guidance, see [ML & DL models](../ml/models_and_best_practices.md) and [Metrics & evaluation](../metrics/metrics_and_evaluation.md).

Mixture analysis estimates component fractions (e.g., EVOO–sunflower blends) from spectra. This workflow uses NNLS when pure references exist and MCR-ALS when they do not.

Suggested visuals: predicted vs true scatter, residual plots, correlation heatmaps for predicted/true fractions. See [Plots guidance](workflow_design_and_reporting.md#plots-visualizations).
For troubleshooting (peak alignment, imbalance of mixtures), see [Common problems & solutions](../troubleshooting/common_problems_and_solutions.md).

```mermaid
flowchart LR
  A[Raw mixtures (+ pure refs)] --> B[Preprocessing (baseline, smoothing, norm, crop)]
  B --> C[Feature space (optional PCA) or stay in spectra]
  C --> D[NNLS (with pure refs) or MCR-ALS (unsupervised)]
  D --> E[Metrics (RMSE, R²), residuals, plots]
  E --> F[Report (pred vs true, residuals, report.md)]
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

## 5. Interpretation (MethodsX tone)
- Compare predicted vs true fractions (if known); report RMSE/MAE and R².
- Inspect residuals; large or structured residuals may indicate missing components or misalignment.
- Main figure: predicted vs true scatter; Supplement: residual plots, spectra overlays.

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

## Further reading
- [Normalization & smoothing](../preprocessing/normalization_smoothing.md)
- [Mixture models & fingerprinting](../ml/mixture_models.md)
- [Model evaluation](../ml/model_evaluation_and_validation.md)
