# Quickstart (Python)

This minimal script loads a spectral library, applies preprocessing, runs PCA and a classifier, and prints metrics.

```python
from pathlib import Path

import matplotlib.pyplot as plt
from foodspec.data import load_library
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.features.ratios import RatioFeatureGenerator
from foodspec.chemometrics.pca import run_pca
from foodspec.chemometrics.models import make_classifier
from foodspec.validation import validate_spectrum_set

# 1) Load library
fs = load_library(Path("libraries/oils_demo.h5"))
validate_spectrum_set(fs)

# 2) Preprocess
preproc = [
    ALSBaseline(lambda_=1e5, p=0.01, max_iter=10),
    SavitzkyGolaySmoother(window_length=9, polyorder=3),
    VectorNormalizer(norm="l2"),
]
X_proc = fs.x
for step in preproc:
    X_proc = step.fit_transform(X_proc)

# 3) Feature: simple ratio (example placeholders)
ratio_gen = RatioFeatureGenerator({"ratio_1655_1742": ("peak_1655.0_height", "peak_1742.0_height")})
# In a real run, feed peak heights; here we keep X_proc as feature matrix for simplicity.

# 4) PCA
_, pca_res = run_pca(X_proc, n_components=2)
plt.figure()
plt.scatter(pca_res.scores[:, 0], pca_res.scores[:, 1], c="steelblue")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
plt.savefig("pca_scores.png", dpi=150)

# 5) Classification
clf = make_classifier("rf", random_state=42)
clf.fit(X_proc, fs.metadata["oil_type"])
acc = clf.score(X_proc, fs.metadata["oil_type"])
print("Training accuracy (demo only):", acc)
```

Notes:
- Replace `libraries/oils_demo.h5` with your own HDF5 library.
- For real ratios, extract peaks first (see Workflows → Oil authentication).
- Save figures/metrics to build your own report.
