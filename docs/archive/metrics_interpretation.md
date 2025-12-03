# Metrics & result interpretation

Questions this page answers
- What do common classification/regression metrics mean?
- How do I compute them on foodspec outputs?
- How should I interpret and report them in food spectroscopy?

## Classification
- **Accuracy**: fraction correct; sensitive to imbalance. Good for balanced datasets; supplement with F1 otherwise.  
- **Precision**: of predicted positives, how many are correct; important when false positives are costly.  
- **Recall**: of true positives, how many were found; important when missing adulterants is costly.  
- **F1-score**: harmonic mean of precision and recall (macro for imbalance).  
- **Confusion matrix**: shows which classes are confused; normalize rows to interpret per-class accuracy.  
- **ROC-AUC**: ranking quality; use when probabilities/scores available.

Guidelines: early/exploratory work may see 0.7–0.85 accuracy/F1; for publication aim higher (≥0.9) on well-designed data; always report class balance and CV design.

### Small code example (classification)
```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# y_true, y_pred from a foodspec workflow (e.g., oil_auth)
acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred, labels=class_labels)
print(acc, f1_macro, cm)
```
Interpretation: combine accuracy with macro/weighted F1 for imbalance; use confusion matrix to inspect per-class errors.

## Regression and mixture analysis
- **RMSE / MAE**: typical absolute error in target units (e.g., fraction or %); smaller is better; relate to acceptable error (e.g., ±0.05 fraction).  
- **R²**: proportion of variance explained; near 1 is good; check residuals for bias.  
- **Bias**: mean error; indicates systematic over/underestimation.  
- **Residuals**: inspect predicted vs true and residual vs true plots.

### Small code example (regression/mixture)
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# y_true, y_pred from mixture/regression workflow
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(rmse, mae, r2)
```
Interpretation: relate RMSE/MAE to allowable error; R² near 1 is good but inspect residuals for structure.

## Cross-validation and robustness
- Use k-fold CV (stratified for classification) and report mean ± std across folds.  
- Consider multiple random seeds for small datasets.  
- Validate on an independent dataset/instrument when possible.

## Reporting
- Main: confusion matrix + macro/weighted F1 (classification); R²/RMSE (regression/mixture).  
- Supplementary: per-class precision/recall/F1, ROC curves, sensitivity analyses, residual plots.  
- Always state preprocessing, features, model type, CV design (folds, stratification), and dataset provenance.

See also
- [stats_tests.md](stats_tests.md)
- [Oil authentication tutorial](../workflows/oil_authentication.md)
- [MethodsX protocol](../protocols/methodsx_protocol.md)
- [API index](../api/index.md)
# Metrics interpretation (legacy)

> **Status:** Legacy/archived. Superseded by the maintained page at `../metrics/metrics_and_evaluation.md`. Kept only for historical reference; use the main metrics chapter for current guidance.
