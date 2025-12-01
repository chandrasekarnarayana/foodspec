# Metrics & result interpretation

## Classification
- **Accuracy**: fraction correct; sensitive to imbalance. Good for balanced datasets; supplement with F1 otherwise.  
- **Precision**: of predicted positives, how many are correct; important when false positives are costly.  
- **Recall**: of true positives, how many were found; important when missing adulterants is costly.  
- **F1-score**: harmonic mean of precision and recall.  
  - **Macro F1**: average across classes equally; good for imbalance.  
  - **Weighted F1**: weighted by class size.  
- **Confusion matrix**: shows which classes are confused; normalize rows to interpret per-class accuracy.
Interpretation in food spectroscopy: minor adulterant classes may be underrepresented; rely on macro F1 and per-class precision/recall to ensure rare classes are detected.

Guidelines (not strict thresholds):
- Exploratory/early work: accuracies/F1 in the 0.7–0.85 range may be acceptable; emphasize limitations.
- Publication/validation: aim for higher (≥0.9) on well-designed datasets; always report class balance and folds.

## Regression and mixture analysis
- **RMSE / MAE**: typical absolute error in target units (e.g., fraction or %). Smaller is better; relate to acceptable error in your application (e.g., ±0.05 fraction).  
- **R²**: proportion of variance explained; near 1 is good, but check residuals for bias.  
- **Bias**: mean error; indicates systematic over/underestimation.  
- **Residuals**: inspect plots of predicted vs true and residual vs true to spot trends.

## Uncertainty and robustness
- Report variability across folds (mean ± std) for CV metrics.  
- Consider multiple random seeds; small datasets can vary a lot.  
- For mixture/regression, include prediction intervals or at least residual distribution summaries.  
- When in doubt, validate on an independent dataset or instrument to check robustness.
