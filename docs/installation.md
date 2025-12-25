!!! warning "Not Canonical — Redirect"
  This page is not the canonical source. Please use [01-getting-started/installation.md](01-getting-started/installation.md).

# Installation

## Requirements
- Python 3.10 or 3.11 (recommended).
- Typical scientific stack: NumPy, SciPy, scikit-learn, pandas, matplotlib, h5py (installed as dependencies).

## User installation
```bash
pip install foodspec
```

Verify:
```bash
foodspec about
```

## Optional extras
- Deep learning (1D CNN prototype):  
  ```bash
  pip install "foodspec[deep]"
  ```  
  Calling `Conv1DSpectrumClassifier` without TensorFlow installed will raise a clear ImportError suggesting this extra.

## Developer installation
```bash
git clone https://github.com/chandrasekarnarayana/foodspec.git
cd foodspec
pip install -e ".[dev]"
```
Run tests to confirm:
```bash
pytest
```
