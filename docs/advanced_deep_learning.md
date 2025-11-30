# Advanced: Deep learning (optional)

Deep learning is **not required** to use foodspec or to reproduce the MethodsX
protocol. However, an experimental 1D CNN classifier is provided for users who
want to explore deep models on spectral data.

## Installation

The deep module depends on TensorFlow and is not installed by default:

```bash
pip install 'foodspec[deep]'
```

If you call `Conv1DSpectrumClassifier` without TensorFlow installed, foodspec
will raise a clear ImportError explaining how to install the extra.

## Conv1DSpectrumClassifier

Example usage (assuming you have installed the deep extra):

```python
import numpy as np
from foodspec.chemometrics.deep import Conv1DSpectrumClassifier
from foodspec.data.loader import load_example_oils

ds = load_example_oils()
clf = Conv1DSpectrumClassifier(epochs=5, batch_size=16)
clf.fit(ds.x, ds.metadata["oil_type"])
proba = clf.predict_proba(ds.x[:2])
pred = clf.predict(ds.x[:2])
print(pred, proba.shape)
```

This class follows an sklearn-like API (fit, predict, predict_proba),
but is intended for experimental use only. For most applications, classical
chemometric models (PLS, SVM, RF, etc.) are recommended.
