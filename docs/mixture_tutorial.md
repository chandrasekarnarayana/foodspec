# Mixture analysis (NNLS / MCR-ALS)

## Problem
Estimate the fraction of component A vs B (e.g., EVOO–sunflower) in a mixture from Raman/FTIR spectra.

## Methods (plain language + math)
- **NNLS (Non-negative least squares)**: solve \( \mathbf{x} \approx \mathbf{S} \cdot \mathbf{c} \) subject to \( c_i \ge 0 \), where \(\mathbf{x}\) is a mixture spectrum, \(\mathbf{S}\) are pure spectra (columns), and \(\mathbf{c}\) are coefficients (fractions).  
- **MCR-ALS (Alternating least squares)**: factorize a matrix of mixtures \( \mathbf{X} \approx \mathbf{C} \mathbf{S}^\top \) with non-negativity; iteratively update concentrations \( \mathbf{C} \) and pure profiles \( \mathbf{S} \).

Assumptions: linear mixing, non-negative spectra and concentrations, shared wavenumber axis.

## Metrics and expectations
- **RMSE / MAE**: error in predicted fraction; small (<0.05–0.1) is good for lab-grade mixtures; industrial contexts may tolerate higher.  
- **R²**: variance explained; closer to 1 indicates better fits.  
- **Bias**: systematic over/underestimation; check residuals vs true fraction.

## CLI examples
- NNLS (single spectrum index):
```bash
foodspec mixture \
  libraries/mixture.h5 \
  --pure-hdf5 libraries/pure_oils.h5 \
  --mode nnls \
  --spectrum-index 0 \
  --output-dir runs/mixture_nnls
```
- MCR-ALS:
```bash
foodspec mixture \
  libraries/mixture.h5 \
  --pure-hdf5 libraries/pure_oils.h5 \
  --mode mcr_als \
  --output-dir runs/mixture_mcr
```
Outputs: coefficients/residuals (NNLS) or C/S matrices (MCR-ALS), metrics.json, plots (optional).

## Python example
```python
from foodspec.chemometrics.mixture import nnls_mixture, mcr_als
coeffs, resid = nnls_mixture(mixture_spectrum, pure_matrix)  # pure_matrix: (n_points, n_components)
C, S = mcr_als(X_mixture, n_components=2)  # X_mixture: (n_samples, n_points)
```

### Quick metric check
```python
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
pred = pure_matrix @ coeffs
print("RMSE:", np.sqrt(mean_squared_error(mixture_spectrum, pred)))
```
Interpretation: small RMSE and high R² indicate good reconstruction; inspect residuals for bias.

## Reporting guidance
- **Main figures**: predicted vs true fraction plot; residual plot.  
- **Supplementary**: recovered pure spectra (S), concentration profiles (C), detailed RMSE/MAE tables.  
- State assumptions (linear mixing, non-negativity) and wavenumber range used; include any calibration/validation split details.

See also
- [metrics_interpretation.md](metrics_interpretation.md)
- [stats_tests.md](stats_tests.md)
- [api_reference.md](api_reference.md)
