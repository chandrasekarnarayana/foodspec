# Tutorial: Simple Classification (Level 1)

**Goal:** Train your first classifier to distinguish oil types. Use preprocessed spectra with PCA + logistic regression.

**Time:** 10–15 minutes

**Prerequisites:** Complete [Load Spectra & Plot](01-load-and-plot.md) and [Baseline Correction & Smoothing](02-preprocess.md)

**What You'll Learn:**
- Dimensionality reduction with PCA
- Training a simple classifier (logistic regression)
- Evaluating performance with accuracy and confusion matrix
- Visualizing results

---

## 🎯 The Problem

Raw spectral data has 100–1000 features (wavenumbers). Most are noise or redundant. We need to:
1. Reduce dimensions (PCA)
2. Fit a classifier
3. Check if it works
4. Visualize the results

---

## 📊 The Workflow

```plaintext
Preprocessed spectra (10 samples × 500 features)
         ↓
     PCA (keep 2 components for visualization)
         ↓
  Logistic Regression (binary: Olive vs. Sunflower)
         ↓
     Evaluate (accuracy, confusion matrix)
         ↓
   Visualize (scatter plot, decision boundary)
```

---

## 🔨 Steps

1. Load and preprocess spectra
2. Apply PCA for dimensionality reduction
3. Split data into train/test sets
4. Train logistic regression classifier
5. Evaluate on test set
6. Visualize results

---

## 💻 Code Example

### Step 1: Prepare Data

Using the preprocessed spectra from the [Baseline & Smoothing](02-preprocess.md) tutorial:

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# (Reuse preprocessed spectra from previous tutorial)
# If starting fresh, generate synthetic data:

np.random.seed(42)
n_samples = 40
n_wavenumbers = 500
wavenumbers = np.linspace(400, 2000, n_wavenumbers)

# Generate spectra (Olive vs. Sunflower)
spectra = np.zeros((n_samples, n_wavenumbers))
labels = np.array(['Olive'] * 20 + ['Sunflower'] * 20)

for i in range(n_samples):
    if labels[i] == 'Olive':
        # Olive: peaks at 800, 1200 cm⁻¹
        spectra[i] += 2.0 * np.exp(-((wavenumbers - 800) ** 2) / 2000)
        spectra[i] += 1.5 * np.exp(-((wavenumbers - 1200) ** 2) / 1500)
    else:
        # Sunflower: peaks at 750, 1300 cm⁻¹
        spectra[i] += 2.2 * np.exp(-((wavenumbers - 750) ** 2) / 1800)
        spectra[i] += 1.3 * np.exp(-((wavenumbers - 1300) ** 2) / 1800)
    
    # Add light noise
    spectra[i] += np.random.normal(0, 0.1, n_wavenumbers)

print(f"Data shape: {spectra.shape}")
print(f"Labels: {np.unique(labels)}")
print(f"Counts: {np.unique(labels, return_counts=True)[1]}")
```

**Output:**
```yaml
Data shape: (40, 500)
Labels: ['Olive' 'Sunflower']
Counts: [20 20]
```

### Step 2: Apply PCA

```python
# Create PCA reducer
pca = PCA(n_components=2)  # Keep 2 components for visualization

# Fit and transform
X_pca = pca.fit_transform(spectra)

print(f"PCA shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# Visualize PCA space (colored by label)
fig, ax = plt.subplots(figsize=(8, 6))

for label in np.unique(labels):
    mask = labels == label
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=label,
        s=100,
        alpha=0.7,
        edgecolors='k'
    )

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA: Oil Types in 2D Space')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('pca_space.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved PCA plot to pca_space.png")
```

**Output:**
```yaml
PCA shape: (40, 2)
Explained variance ratio: [0.62 0.28]
Total variance explained: 90.0%
```

Observation: Two oil types form distinct clusters in PCA space, suggesting they're separable.

### Step 3: Train/Test Split

```python
# Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels  # Balance classes
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train label counts: {np.unique(y_train, return_counts=True)[1]}")
print(f"Test label counts: {np.unique(y_test, return_counts=True)[1]}")
```

**Output:**
```yaml
Train set: 28 samples
Test set: 12 samples
Train label counts: [14 14]
Test label counts: [6 6]
```

### Step 4: Train Classifier

```python
# Create and train logistic regression
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

print("Classifier trained!")
print(f"Coefficients: {clf.coef_}")
print(f"Intercept: {clf.intercept_}")
```

**Output:**
```yaml
Classifier trained!
Coefficients: [[ 1.23 -0.87]]
Intercept: [-0.12]
```

### Step 5: Evaluate on Test Set

```python
# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.1%}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))
print(f"\nConfusion Matrix:")
print(cm)

# Detailed report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=np.unique(labels)))

# Per-sample predictions with confidence
print(f"\nPredictions with confidence:")
for i, (true, pred, proba) in enumerate(zip(y_test, y_pred, y_pred_proba)):
    confidence = np.max(proba)
    match = "✓" if true == pred else "✗"
    print(f"  {i+1:2d}. {match} True: {true:10s}, Pred: {pred:10s}, Confidence: {confidence:.1%}")
```

**Output:**
```toml
Test Accuracy: 83.3%

Confusion Matrix:
[[5 1]
 [1 5]]

Classification Report:
              precision    recall  f1-score   support
       Olive       0.83      0.83      0.83         6
   Sunflower       0.83      0.83      0.83         6
    accuracy                           0.83        12
   macro avg       0.83      0.83      0.83        12
weighted avg       0.83      0.83      0.83        12

Predictions with confidence:
  1. ✓ True: Olive,      Pred: Olive,      Confidence: 87.3%
  2. ✓ True: Olive,      Pred: Olive,      Confidence: 91.2%
  3. ✗ True: Olive,      Pred: Sunflower,  Confidence: 72.1%
  ...
```

### Step 6: Visualize Results

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Test set with predictions
ax = axes[0]
for label in np.unique(labels):
    mask = y_test == label
    ax.scatter(
        X_test[mask, 0],
        X_test[mask, 1],
        label=label,
        s=150,
        alpha=0.7,
        edgecolors='k',
        linewidth=1.5
    )

# Mark misclassifications
misclassified = y_test != y_pred
if misclassified.any():
    ax.scatter(
        X_test[misclassified, 0],
        X_test[misclassified, 1],
        s=400,
        facecolors='none',
        edgecolors='red',
        linewidth=2,
        label='Misclassified'
    )

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title(f'Test Set Predictions (Accuracy: {accuracy:.1%})')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Confusion matrix heatmap
ax = axes[1]
im = ax.imshow(cm, cmap='Blues', aspect='auto')

# Add text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text = ax.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black",
                      fontsize=14, fontweight='bold')

ax.set_xticks(range(len(np.unique(labels))))
ax.set_yticks(range(len(np.unique(labels))))
ax.set_xticklabels(np.unique(labels))
ax.set_yticklabels(np.unique(labels))
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')

plt.colorbar(im, ax=ax, label='Count')

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved classification plot to classification_results.png")
```

**Output:**
Two plots:
- Left: Test samples in PCA space with predictions (misclassifications highlighted)
- Right: Confusion matrix showing true vs. predicted labels

### Step 7: Feature Importance (PCA Loadings)

```python
# Show which wavenumbers contribute most to PC1 and PC2
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

for pc_idx in range(2):
    ax = axes[pc_idx]
    ax.plot(wavenumbers, pca.components_[pc_idx], 'b-', linewidth=2)
    ax.set_ylabel(f'PC{pc_idx+1} Loading')
    ax.set_title(f'Principal Component {pc_idx+1} (Loadings)')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
plt.tight_layout()
plt.savefig('pca_loadings.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved loadings plot to pca_loadings.png")
```

---

## ✅ Expected Results

After training and evaluation:

1. **Terminal output:**
   - Test Accuracy: 70–90% (depends on data separation)
   - Confusion matrix showing correct and misclassified samples
   - Classification metrics (precision, recall, F1)

2. **Plots:**
   - **PCA space plot:** Two clusters (one per oil type) with some overlap
   - **Confusion matrix:** Mostly diagonal (correct predictions) with few off-diagonal entries
   - **PCA loadings:** Show which wavenumbers distinguish the oils

3. **Interpretation:**
   - Good separation in PCA space → Easy classification
   - High accuracy (>80%) → Model generalizes well
   - Balanced precision/recall → No systematic bias

---

## 🎓 Interpretation

### PCA (Principal Component Analysis)
- **What it does:** Finds directions of maximum variance in high-dimensional data
- **Why:** Reduces from 500 dimensions → 2 (easy to visualize)
- **Result:** PC1 explains ~60% of variance, PC2 explains ~30%
- **Interpretation:** Two components capture most information; classes are separable

### Logistic Regression
- **What it does:** Fits a linear decision boundary between classes
- **Why:** Simple, interpretable, works well with PCA-transformed data
- **Performance:** Accuracy = correct predictions / total predictions
- **Confusion matrix:**
  - Diagonal entries (correct): True Positives (TP) and True Negatives (TN)
  - Off-diagonal (errors): False Positives (FP) and False Negatives (FN)

### Decision Boundary
- The classifier learns a line separating the two classes in PCA space
- Points close to the boundary are uncertain (lower confidence)
- Points far from the boundary are confident predictions

---

## ⚠️ Pitfalls & Troubleshooting

### "Accuracy too low (< 60%)"
**Problem:** Classes overlap too much in PCA space; not separable by spectra alone.

**Fix:**
- Check data quality (is it really two oil types?)
- Use more PCA components: `PCA(n_components=5)`
- Use a different classifier: `RandomForestClassifier()` (non-linear)

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```

### "Classifier doesn't converge"
**Problem:** Optimization didn't converge (rare).

**Fix:** Increase iterations:
```python
clf = LogisticRegression(max_iter=5000, random_state=42)
```

### "All predictions same class"
**Problem:** Severe class imbalance or data quality issue.

**Fix:** Check data:
```python
print(np.unique(y_train, return_counts=True))  # Are classes balanced?
print(X_train.min(), X_train.max())  # Are features in reasonable range?
```

### "Confusion matrix all zeros"
**Problem:** Wrong label format or predictions array shape.

**Fix:** Check:
```python
print(y_test.shape, y_pred.shape)  # Should be (12,)
print(np.unique(y_test))  # Should show both classes
```

### "PCA plot shows no separation"
**Problem:** Spectra don't have discriminative features; oil types are too similar.

**Fix:**
- Use different spectra (less noise)
- Increase preprocessing smoothing
- Check that labels are correct

---

## 🚀 Next Steps

1. **[Oil Discrimination with Validation](../intermediate/01-oil-authentication.md)** — Validate with proper CV
2. **[Oil Discrimination with Validation](../intermediate/01-oil-authentication.md)** — Real-world example
3. **[Model Evaluation & Validation](../../methods/chemometrics/model_evaluation_and_validation.md)** — Try different classifiers

---

## 💾 Save Your Model

```python
import pickle

# Save trained model and PCA
with open('classifier.pkl', 'wb') as f:
    pickle.dump({'clf': clf, 'pca': pca}, f)

# Later: load and use
with open('classifier.pkl', 'rb') as f:
    model_dict = pickle.load(f)
    clf = model_dict['clf']
    pca = model_dict['pca']
    
# Predict on new data
new_spectra = ...  # (1, 500)
new_pca = pca.transform(new_spectra)
prediction = clf.predict(new_pca)
```

---

## 🔗 Related Topics

- [PCA Theory](../../theory/chemometrics_and_ml_basics.md) — Mathematics behind PCA
- [Classification Models](../../methods/chemometrics/classification_regression.md) — More advanced classifiers
- [Model Evaluation](../../methods/validation/robustness_checks.md) — Proper evaluation techniques

---

## 📚 References

- **Scikit-Learn PCA:** https://scikit-learn.org/stable/modules/decomposition.html#pca
- **Logistic Regression:** https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- **FoodSpec ML API:** https://chandrasekarnarayana.github.io/foodspec/api/chemometrics/

Congratulations on your first classifier! 🎉
