# Model Interpretability in FoodSpec

Questions this page answers:
- How can I interpret ML/DL models applied to spectra?
- Which tools does FoodSpec provide for feature importance and latent factors?
- How should interpretability be reported in food spectroscopy studies?

## Why interpretability matters
- Connects spectral features to chemistry (e.g., unsaturation bands).
- Builds trust for QA/QC and regulatory contexts.
- Helps diagnose spurious correlations or leakage.

## Tools in FoodSpec
- **PCA/PLS loadings:** via `plot_pca_loadings` and PLS loadings to identify influential wavenumbers/peaks.
- **Random Forest feature importances:** inspect `feature_importances_`; relate top bands to chemistry.
- **Peak/ratio-based features:** inherently interpretable; report definitions and effect directions.
- **Confusion matrices and per-class metrics:** clarify where models struggle.
- **Residuals and calibration plots:** show bias and spread in regression.

### Reading loadings (PCA/PLS)
- **Scores vs loadings:** scores plot shows samples in latent space; loadings plot shows which wavenumbers drive each component.
- **Example (oils):** If oil A vs B separate along PC1 and loadings have strong positive contributions near ~1655 cm⁻¹ (unsaturation), that band is characteristic of oil A. Use [Spectroscopy basics](../foundations/spectroscopy_basics.md) for vibrational assignments.
- Pair qualitative plots with metrics (e.g., silhouette/between-within on scores) to quantify separation.

## Practical examples
```python
from foodspec.chemometrics.pca import run_pca
from foodspec.viz import plot_pca_loadings

pca, res = run_pca(X, n_components=2)
plot_pca_loadings(res.loadings, wavenumbers, components=(1, 2))
```

```python
# RF feature importances
rf = make_classifier("rf", n_estimators=200, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
```

For PLS-DA/PLS regression, examine loading vectors and VIP-like interpretations; ensure preprocessing is consistent across train/test.

## Visual examples
![Random Forest feature importances](../assets/rf_feature_importance.png)
*Figure: RF feature importances on synthetic spectra. Top bands correspond to simulated peaks; relate them to known chemical bands. Flat or noisy importances may indicate weak signal or overfitting.*

![PLS loadings example](../assets/pls_loadings.png)
*Figure: PLS loadings for the first component, showing influential wavenumbers in a calibration task. Loadings sign shows positive/negative association with the component; peaks in loadings point to bands driving separation or prediction.*

## Reporting guidance
- Report top contributing bands/ratios and their chemical meaning.
- Include loadings/importance plots in supplementary material; summarize key drivers in main text.
- Avoid over-interpretation of noisy or collinear features; cross-check with domain knowledge.
- Pair interpretability with metrics: a model that performs poorly but shows plausible bands still needs improved performance.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for feature importance and model interpretability:**

1. **Feature importance from overfit model (high training accuracy but low test accuracy)**
   - Importance scores reflect noise fit during overfitting, not true signal
   - Top features may be coincidental collinearities in training data
   - **Fix:** Report feature importance from test set or cross-validation; use permutation importance; validate feature relevance independently

2. **Correlations with labels misinterpreted as feature importance**
   - High correlation doesn't mean feature is used by model or is important for prediction
   - Spurious correlations (batch confounding, data leakage) can rank high
   - **Fix:** Use model-specific importance (coefficients, tree splits, SHAP values); don't rely on univariate correlations alone

3. **Loadings/coefficients from collinear features not interpreted cautiously**
   - High collinearity (VIF > 10) makes coefficients unstable; small data changes flip signs
   - Ranking features by coefficient magnitude misleads when collinearity present
   - **Fix:** Check VIF or condition number; use regularization (Ridge/Lasso) to stabilize coefficients; use permutation importance

4. **Confidence intervals not reported for feature importance**
   - Point estimates of importance vary across CV folds; single-fold importance is unstable
   - Without CI, importance scores appear more certain than they are
   - **Fix:** Compute importance across CV folds; report mean ± SD or 95% CI; visualize variability

5. **Statistical significance confused with practical importance (p < 0.05 band ≠ important for prediction)**
   - ANOVA or t-test on feature values tests association, not model utility
   - Significant association doesn't guarantee predictive power
   - **Fix:** Report model-based importance; tie to prediction accuracy; validate with independent data

6. **Interpreting absence of feature as "not important" (feature has importance = 0)**
   - Feature with zero importance may be redundant with other features, not truly unimportant
   - Or importance measure is insensitive to that feature's contribution
   - **Fix:** Use multiple importance measures (permutation, SHAP, coefficients); test feature removal effect on metrics

7. **Chemically implausible features ranked as most important without flagging as suspicious**
   - Food spectra are noisy; random noise peaks can rank highly if model overfit
   - Chemical bands should make physical/chemical sense
   - **Fix:** Cross-check with domain expertise; validate top features in independent data; avoid over-interpreting noisy regions

8. **Feature importance from single model type only (only PLS loadings, no RF permutation importance)**
   - Different models and importance measures can rank features differently
   - Robust conclusions require agreement across multiple approaches
   - **Fix:** Report importance from multiple models/methods; highlight consensus features; note disagreements

## See also
- Metrics & evaluation: [metrics_and_evaluation](../metrics/metrics_and_evaluation.md)
- ML models & best practices: [models_and_best_practices](models_and_best_practices.md)
- Stats: [ANOVA & MANOVA](../stats/anova_and_manova.md) for group-level effects
- Troubleshooting: [common_problems_and_solutions](../troubleshooting/common_problems_and_solutions.md)
