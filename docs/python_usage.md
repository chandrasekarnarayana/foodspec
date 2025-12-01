# Python API usage

Use foodspec as a library to build custom pipelines.

## Load data
```python
from foodspec.data import load_library
fs = load_library("libraries/oils_demo.h5")
```

## Preprocess
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

## Features & models
```python
from foodspec.features.peaks import PeakFeatureExtractor
from foodspec.chemometrics.models import make_classifier
from foodspec.chemometrics.validation import compute_classification_metrics

peaks = PeakFeatureExtractor(expected_peaks=[1655, 1742], tolerance=8.0)
feat = peaks.fit_transform(X, wavenumbers=fs.wavenumbers)
clf = make_classifier("rf", random_state=0)
clf.fit(feat, fs.metadata["oil_type"])
pred = clf.predict(feat)
print(compute_classification_metrics(fs.metadata["oil_type"], pred))
```

## Visualization
```python
import matplotlib.pyplot as plt
from foodspec.viz.pca import plot_pca_scores
from foodspec.chemometrics.pca import run_pca

_, res = run_pca(X, n_components=2)
fig, ax = plt.subplots()
plot_pca_scores(res.scores, labels=fs.metadata["oil_type"], ax=ax)
plt.savefig("pca.png", dpi=150)
```

Adapt these steps for heating, mixture analysis, or QC by swapping feature/model blocks.
