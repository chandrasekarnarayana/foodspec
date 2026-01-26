"""
FoodSpec Professional Chemometrics Core Documentation
======================================================

Phase 1 Implementation - Complete Feature Set
"""

# Chemometrics Core Features Overview

## 1. Spectral Alignment

### Cross-Correlation Alignment
Aligns spectra using FFT-based cross-correlation to correct for instrumental drift or peak shifts.

**Usage:**
```python
from foodspec.features.alignment import align_spectra

X_aligned = align_spectra(X_raw, method="xcorr", max_shift=50, reference_idx=0)
```

**Parameters:**
- `method`: "xcorr" or "dtw"
- `max_shift`: Maximum allowed shift (default: 50)
- `reference_idx`: Reference spectrum index (default: 0)

**Applications:**
- Correcting for instrument drift
- Aligning replicate measurements
- Handling spectral baseline shifts

### Dynamic Time Warping (DTW)
Aligns spectra using DTW with Sakoe-Chiba band windowing for non-linear warping.

**Usage:**
```python
X_aligned = align_spectra(X_raw, method="dtw", window=50)
```

**Features:**
- Sakoe-Chiba band for computational efficiency
- Automatic warping path computation
- Spectral resampling along optimal path

**Applications:**
- Handling temperature-induced peak shifts
- Processing spectra from different instruments
- Resolving non-linear peak distortions


## 2. NNLS Spectral Unmixing

Non-negative least squares unmixing for determining pure component concentrations in mixtures.

**Usage:**
```python
from foodspec.features.unmixing import NNLSUnmixer, unmix_spectrum

# Using class API
unmixer = NNLSUnmixer()
unmixer.fit(pure_component_library)  # Shape: (n_components, n_wavenumbers)
concentrations = unmixer.transform(mixture_spectra)

# Using function API
concentrations = unmix_spectrum(mixtures, library)
```

**Features:**
- Constrained non-negative optimization
- Residual computation
- Reconstruction capability
- Batch processing support

**Applications:**
- Quantifying compound concentrations
- Quality control (QC) of food products
- Adulterant detection
- Composition analysis


## 3. Partial Least Squares Regression (PLSR)

PLSR with Variable Importance in Projection (VIP) scores for feature selection.

**Usage:**
```python
from foodspec.modeling.chemometrics import PLSRegression, VIPCalculator

# PLSR model
pls = PLSRegression(n_components=5, scale=True)
pls.fit(X_train, y_train)
y_pred = pls.predict(X_test)

# VIP scores
vip = VIPCalculator.calculate_vip(X, y, n_components=5)
VIPCalculator.plot_vip(vip, feature_names=feature_names)
```

**Features:**
- NIPALS algorithm for component extraction
- Cross-validated component selection
- VIP scores for interpretability
- Automatic scaling

**Applications:**
- Regression on spectral data
- Feature importance ranking
- Dimensionality reduction
- Prediction with few samples


## 4. Non-Negative Regression (NNLS)

NNLS and Constrained LASSO for regression with non-negativity constraints.

**Usage:**
```python
from foodspec.modeling.chemometrics import NNLSRegression, ConstrainedLasso

# NNLS Regression
nnls_reg = NNLSRegression(scale=True)
nnls_reg.fit(X_train, y_train)
y_pred = nnls_reg.predict(X_test)

# Constrained LASSO (with sparsity)
classo = ConstrainedLasso(alpha=0.01, sum_to_one=False)
classo.fit(X_train, y_train)
sparsity = classo.sparsity()  # Fraction of zero coefficients
```

**Features:**
- Guaranteed non-negative coefficients
- Optional sum-to-one constraint (for compositional data)
- Sparsity via LASSO
- Residual analysis

**Applications:**
- Abundance estimation
- Compositional data analysis
- Forced non-negativity regression
- Sparse mixture modeling


## 5. Bootstrap Stability Analysis

Statistical framework for assessing model parameter and prediction stability.

**Usage:**
```python
from foodspec.validation.stability import BootstrapStability, StabilityIndex
from sklearn.linear_model import Ridge

bs = BootstrapStability(n_bootstrap=100, confidence=0.95)

# Parameter stability
param_mean, param_std, param_ci = bs.assess_parameter_stability(
    X, y,
    fit_func=lambda x, yy: Ridge().fit(x, yy),
    param_func=lambda m: m.coef_
)

# Prediction stability
pred_mean, pred_std, pred_ci = bs.assess_prediction_stability(
    X_train, y_train,
    fit_func=lambda x, yy: Ridge().fit(x, yy),
    X_test=X_test
)

# Jackknife resampling
jk_mean, jk_std = StabilityIndex.jackknife_resampling(
    X, y,
    fit_func=lambda x, yy: Ridge().fit(x, yy),
    param_func=lambda m: m.coef_
)
```

**Features:**
- Confidence interval computation
- Jackknife resampling
- Reproducibility indices
- Sensitivity analysis

**Applications:**
- Model validation
- Uncertainty quantification
- Parameter reliability assessment
- Robustness to sampling variation


## 6. Agreement Analysis

Statistical methods for comparing two measurement methods:
- **Bland-Altman Analysis**: Visualizes systematic and random agreement
- **Deming Regression**: Accounts for error in both variables

**Usage:**
```python
from foodspec.validation.agreement import BlandAltmanAnalysis, DemingRegression

# Bland-Altman
ba = BlandAltmanAnalysis(confidence=0.95)
mean_diff, std_diff, ll, ul, corr = ba.calculate(method1, method2)
ba.plot()  # Generate Bland-Altman plot
report = ba.get_report()

# Deming Regression
deming = DemingRegression(variance_ratio=1.0)
deming.fit(reference_method, test_method)
y_pred = deming.predict(reference_method)
ccc = deming.get_concordance_correlation(reference_method, test_method)
deming.plot()
```

**Features:**
- Limits of agreement (LOA)
- Concordance correlation coefficient (CCC)
- Perpendicular residuals
- Systematic vs random error quantification

**Applications:**
- Instrument intercomparison
- Method validation
- QC system evaluation
- Reference method establishment


## 7. Drift Monitoring (EWMA)

Exponentially Weighted Moving Average (EWMA) control charts and drift detection.

**Usage:**
```python
from foodspec.qc.drift_ewma import EWMAControlChart, DriftDetector

# EWMA Control Chart
ewma = EWMAControlChart(lambda_=0.2, confidence=0.99)
ewma.initialize(X_reference)  # Reference/calibration data

# Monitor single observation
ewma_val, is_alarm = ewma.update(x_new)

# Monitor stream
X_stream = np.random.randn(50, 3)
ewma_vals, alarms = ewma.process(X_stream)
ewma.plot()  # Control chart visualization

# Comprehensive drift detection
dd = DriftDetector(lambda_=0.2)
dd.initialize(X_reference)
result = dd.check_drift(x_new)  # Single observation
results = dd.process_stream(X_stream)  # Stream processing
summary = dd.get_drift_summary()
fig = dd.plot_drift_report()
```

**Features:**
- EWMA charts for mean drift detection
- Mahalanobis distance for multivariate outliers
- Chi-square threshold for outlier identification
- Multi-panel drift report visualization

**Applications:**
- Real-time instrument monitoring
- Drift early warning
- QC system health tracking
- Out-of-specification detection
- Data quality assessment


## Integration with Protocol System

All chemometrics features integrate with FoodSpec's protocol system for YAML-based workflow definition.

### Example Protocol Configuration

```yaml
steps:
  - name: alignment
    type: alignment
    params:
      method: dtw
      window: 50
      reference_idx: 0
  
  - name: unmixing
    type: unmixing
    params:
      library_path: lib_pure_components.csv
  
  - name: plsr
    type: plsr
    params:
      n_components: 5
      scale: true
  
  - name: drift_monitor
    type: drift_monitor
    params:
      lambda: 0.2
      confidence: 0.99
```


## Performance Characteristics

### Computational Complexity

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| Cross-Correlation | O(n log n) | O(n) | FFT-based, very fast |
| DTW | O(nm²) | O(nm) | Sakoe-Chiba band reduces complexity |
| NNLS | O(kn²) | O(n) | k = iterations, typically <10 |
| PLSR | O(pqr) | O(pq) | p=samples, q=features, r=components |
| Bootstrap | O(B × m) | O(m) | B=bootstrap samples, linear in B |
| EWMA | O(1) | O(1) | Online/streaming |

### Convergence & Accuracy

- **PLSR NIPALS**: Converges in <20 iterations for typical data
- **NNLS**: Solves exactly (guaranteed non-negative solution)
- **DTW**: Exact optimal path computation
- **Bootstrap**: Statistical accuracy improves with B; 100 samples sufficient for CI estimation


## Quality Assurance & Testing

All modules include:
- Comprehensive unit tests (34 tests total)
- Integration tests
- Sklearn compatibility validation
- Edge case handling
- Numerical stability checks

**Test Coverage:**
- Cross-correlation alignment: 2 tests
- DTW alignment: 2 tests
- NNLS unmixing: 3 tests
- PLSR: 3 tests
- NNLS regression: 2 tests
- Constrained LASSO: 2 tests
- Bootstrap stability: 2 tests
- Stability indices: 3 tests
- Bland-Altman analysis: 2 tests
- Deming regression: 2 tests
- EWMA control chart: 3 tests
- Drift detector: 4 tests
- Integration tests: 4 tests


## Best Practices

1. **Always center and scale** spectral data before analysis
2. **Use bootstrap stability** to assess model reliability
3. **Validate instrument agreement** with Bland-Altman before deployment
4. **Monitor drift continuously** during operation
5. **Verify alignment quality** visually before downstream analysis
6. **Assess unmixing residuals** for mixture complexity detection
7. **Use cross-validation** with PLSR for component selection


## Related Documentation

- [Alignment Methods](../tutorials/spectral_alignment.md)
- [Unmixing Applications](../tutorials/spectral_unmixing.md)
- [PLSR Guide](../tutorials/plsr_vip.md)
- [Stability Analysis](../tutorials/bootstrap_stability.md)
- [Agreement Analysis](../tutorials/agreement_analysis.md)
- [QC Systems](../qc_system.md)
- [Protocol System](../workflows/protocols.md)


## References

### Alignment
- Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition
- Salvador, S., & Chan, P. (2007). DTW distance: Practical considerations

### PLSR & VIP
- Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics
- Xia, W., et al. (2019). A review on variable selection methods in multivariate analysis

### Agreement Analysis
- Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods
- Lin, L. I. (1989). A concordance correlation coefficient to evaluate reproducibility
- Deming, W. E. (1943). Statistical adjustment of data

### Bootstrap & Stability
- Efron, B., & Tibshirani, R. (1993). An introduction to the bootstrap
- Miller, R. G. (1974). The jackknife - a review

### EWMA Control Charts
- Roberts, S. W. (1959). Control chart tests based on geometric moving averages
- Montgomery, D. C. (2012). Introduction to statistical quality control
