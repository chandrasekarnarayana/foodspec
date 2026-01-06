# Oil Authentication: Supervised Classification

**Level**: Beginner → Intermediate  
**Runtime**: ~10 seconds  
**Key Concepts**: Classification, cross-validation, confusion matrices, model discrimination

---

## What You Will Learn

In this example, you'll learn how to:
- Load spectroscopy data from CSV files
- Train a classifier to distinguish oils by type
- Evaluate model performance with cross-validation
- Interpret confusion matrices and classification metrics
- Visualize data structure using dimensionality reduction (PCA)

After completing this example, you'll understand the workflow for any classification problem in FoodSpec (fraud detection, quality assessment, authenticity verification).

---

## Prerequisites

- Basic Python knowledge
- Familiarity with NumPy arrays and Pandas DataFrames
- Understanding of supervised learning concepts (train/test, classification)
- `numpy`, `pandas`, `matplotlib`, `scikit-learn` installed

**Optional background**: Read [Chemometrics & ML Basics](../theory/chemometrics_and_ml_basics.md)

---

## The Problem

**Real-world scenario**: You're a food manufacturer testing whether your olive oil supplies are authentic. You have reference spectra for virgin olive oil, processed olive oil, and two adulterants (sunflower, canola). Can you build a classifier to automatically detect fake oils?

**Data**: Raman spectra (intensity vs. wavenumber) for 8 samples across 4 classes.

**Goal**: Train a model, evaluate with cross-validation, interpret results.

---

## Step 1: Load Data

```python
import numpy as np
import pandas as pd
from pathlib import Path

# Load synthetic oil spectra (8 samples × 1500 wavelengths)
data = pd.read_csv("examples/data/oil_synthetic.csv", index_col=0)
X = data.drop("OilType", axis=1).values
y = data["OilType"].values

print(f"Data shape: {X.shape}")  # (8, 1500)
print(f"Oil types: {np.unique(y)}")  # ['CanOil' 'OliveOil' 'ProcessedOl' 'SunflowerOil']
```

**What's happening**:
- The CSV contains 8 spectra and their oil type labels
- `X` contains the spectral intensities (predictors)
- `y` contains the oil type labels (targets)

---

## Step 2: Train with Cross-Validation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Initialize classifier
clf = RandomForestClassifier(n_estimators=50, random_state=42)

# Evaluate with 5-fold cross-validation
scores = cross_validate(
    clf, X, y, 
    cv=5, 
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
)

print(f"CV Accuracy: {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
print(f"CV F1: {scores['test_f1_macro'].mean():.3f} ± {scores['test_f1_macro'].std():.3f}")
```

**What's happening**:
- RandomForestClassifier is trained on 4/5 of the data, tested on 1/5
- This repeats 5 times, rotating which fold is held out
- We compute accuracy, precision, recall, F1 across all folds
- Metrics near 1.0 indicate good discrimination

---

## Step 3: Visualize Performance & Structure

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Get predictions from cross-validation
y_pred = cross_val_predict(clf, X, y, cv=5)
cm = confusion_matrix(y, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(np.unique(y))))
ax.set_yticks(range(len(np.unique(y))))
ax.set_xticklabels(np.unique(y), rotation=45)
ax.set_yticklabels(np.unique(y))
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Oil Classification: Confusion Matrix")

# Add counts
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w")

plt.tight_layout()
plt.savefig("oil_auth_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Interpretation**:
- Diagonal elements = correct classifications
- Off-diagonal = misclassifications
- Perfect classifier = only non-zero diagonals
- This plot reveals which oils are confused with others

---

## Full Working Script

See the production script with enhanced documentation, output directory management, and additional analysis:

📄 **[`examples/oil_authentication_quickstart.py`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/oil_authentication_quickstart.py)** – Full working code (35 lines)

---

## Generated Figure

![Confusion Matrix](https://github.com/chandrasekarnarayana/foodspec/raw/mahttps://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/oil_auth_confusion.png)

**Figure interpretation**:
- Rows = true oil types
- Columns = predicted oil types
- Perfect classification: all counts on diagonal
- Model performance: assess which oils are well-distinguished

---

## Key Takeaways

✅ **Classification workflow**: Load → Train → Cross-validate → Evaluate  
✅ **Cross-validation**: Prevents overfitting by rotating train/test splits  
✅ **Confusion matrix**: Shows misclassification patterns, not just accuracy  
✅ **Metrics matter**: Precision/recall assess class-specific performance  

---

## Real-World Applications

- 🌾 **Olive oil authentication**: Detect counterfeit high-value oils
- 🍯 **Honey fraud detection**: Distinguish pure from adulterated honey
- 🧈 **Butter authenticity**: Identify margarine substitution
- 🥛 **Milk origin verification**: Grass vs. grain-fed dairy

---

## Next Steps

1. **Try it**: Modify the classifier (e.g., use SVM, Logistic Regression)
2. **Explore**: Change cross-validation folds (cv=10) and observe variance
3. **Learn more**: Read [Classification & Regression](../methods/chemometrics/classification_regression.md)
4. **Advance**: See [Oil Authentication Workflow](../workflows/authentication/oil_authentication.md) for complete domain example

---

## Interactive Notebook

For step-by-step exploration with visualizations:

📓 **[`examples/tutorials/01_oil_authentication_teaching.ipynb`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/01_oil_authentication_teaching.ipynb)**

---

## Figure provenance
- Generated by [scripts/generate_docs_figures.py](https://github.com/chandrasekarnarayana/foodspec/blob/main/scripts/generate_docs_figures.py)
- Output: [../assets/figures/oil_confusion.png](../assets/figures/oil_confusion.png)

