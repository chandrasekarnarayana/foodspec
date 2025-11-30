# Oil Authentication Tutorial

This tutorial walks through the end-to-end oil authentication workflow using **foodspec**.

## 1. Load data
```python
from foodspec.data.loader import load_example_oils

spectra = load_example_oils()
print(spectra.x.shape, spectra.metadata.head())
```

## 2. Inspect raw spectra
```python
from foodspec.viz.spectra import plot_spectra, plot_mean_spectrum
import matplotlib.pyplot as plt

plot_spectra(spectra, color_by="oil_type")
plt.show()
```

## 3. Run the authentication workflow
```python
from foodspec.apps.oils import run_oil_authentication_workflow

result = run_oil_authentication_workflow(spectra, label_column="oil_type", classifier_name="rf", cv_splits=5)
print(result.cv_metrics)
print("Confusion matrix:\n", result.confusion_matrix)

# Inspect feature importances (if available, e.g., RandomForest)
if result.feature_importances is not None:
    print(result.feature_importances.sort_values(ascending=False).head())
```

## 4. Visualize results
```python
from foodspec.viz.classification import plot_confusion_matrix

plot_confusion_matrix(result.confusion_matrix, class_names=result.class_labels)
plt.show()
```

## 5. Inspect features and model
- `result.pipeline` holds the fitted preprocessing + feature + classifier pipeline.
- `result.feature_importances` (if available) ranks important peaks/ratios.
- Change classifier by setting `classifier_name` (e.g., `"rf"`, `"svm_rbf"`, `"logreg"`) in the workflow or CLI options.

## 6. Next steps
- Swap classifiers via `classifier_name` (e.g., `svm_rbf`, `logreg`, `rf`).
- Integrate your own datasets using `load_folder` and consistent metadata columns.
