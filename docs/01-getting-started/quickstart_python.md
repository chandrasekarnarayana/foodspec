# Quickstart (Python API)

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Python developers; data scientists; researchers needing programmatic control and custom workflows.
**What problem does this solve?** Using FoodSpec as a Python library for custom analysis pipelines and integration.
**When to use this?** When you need fine-grained control; building custom workflows; integrating with existing Python code.
**Why it matters?** Python API provides maximum flexibility for custom preprocessing, feature engineering, and model development.
**Time to complete:** 15-20 minutes.
**Prerequisites:** FoodSpec installed; Python 3.10+; Jupyter or Python environment; basic pandas/scikit-learn knowledge

---

This walkthrough shows the main steps: load data, preprocess, explore with PCA, and run a classifier. Replace paths/labels with your own.

## 1) Load and validate
```python
from pathlib import Path
from foodspec import load_library
from foodspec.validation import validate_spectrum_set

fs = load_library(Path("libraries/oils_demo.h5"))
validate_spectrum_set(fs)
```

## 2) Preprocess (baseline + smoothing + normalization)
```python
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import VectorNormalizer

X = fs.x
for step in [
    ALSBaseline(lambda_=1e5, p=0.01, max_iter=10),
    SavitzkyGolaySmoother(window_length=9, polyorder=3),
    VectorNormalizer(norm="l2"),
]:
    X = step.fit_transform(X)
```

## 3) Explore with PCA
```python
import matplotlib.pyplot as plt
from foodspec.chemometrics.pca import run_pca

_, pca_res = run_pca(X, n_components=2)
plt.scatter(pca_res.scores[:, 0], pca_res.scores[:, 1], c="steelblue")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
plt.savefig("pca_scores.png", dpi=150)
```

## 4) Train a quick classifier (demo)
```python
from foodspec.chemometrics.models import make_classifier

clf = make_classifier("rf", random_state=42)
clf.fit(X, fs.metadata["oil_type"])
acc = clf.score(X, fs.metadata["oil_type"])
print("Training accuracy (demo only):", acc)
```

Notes:
- Replace `libraries/oils_demo.h5` and `oil_type` with your own library/label.
- For real features, extract peaks/ratios first (see oil-auth workflow); the above uses preprocessed spectra directly for brevity.
- Save plots/metrics for reporting.
