# Mixture Analysis (Teaching Walkthrough)

**Focus**: NNLS unmixing • Quantification • Residuals

## What you will learn
- Build pure component spectra and synthetic mixtures
- Use Non-Negative Least Squares to estimate component fractions
- Check fit quality with residuals and simple error metrics

## Prerequisites
- Python 3.10+
- numpy, scipy

## Minimal runnable code
```python
import numpy as np
from scipy.optimize import nnls

np.random.seed(42)
w = np.linspace(800, 3000, 1500)
comp1 = np.exp(-((w-1600)/200)**2)
comp2 = np.exp(-((w-1500)/150)**2)
true = np.array([0.7, 0.3])
mixture = true[0]*comp1 + true[1]*comp2 + 0.01*np.random.randn(len(w))

A = np.column_stack([comp1, comp2])
est, res = nnls(A, mixture)
est /= est.sum()
mae = np.abs(est-true).mean()
print({"estimated": est.tolist(), "mae": float(mae), "rmse": float(np.sqrt(res))})
```

## Explain the outputs
- `estimated` fractions should be close to `[0.7, 0.3]`
- `mae` / `rmse` quantify unmixing accuracy
- Residuals near noise level ⇒ linear model is appropriate

## Full resources
- Full script: https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/mixture_analysis_quickstart.py
- Teaching notebook (download/run): https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/03_mixture_analysis_teaching.ipynb

## Run it yourself
```bash
python examples/mixture_analysis_quickstart.py
jupyter notebook examples/tutorials/03_mixture_analysis_teaching.ipynb
```

## Related docs
- Methods: mixture models → ../methods/chemometrics/mixture_models.md
- Workflow: mixture analysis → ../workflows/quantification/mixture_analysis.md
