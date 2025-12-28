# Installation

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Anyone wanting to install FoodSpec for food spectroscopy analysis.  
**What problem does this solve?** Setting up FoodSpec and its dependencies correctly.  
**When to use this?** First-time installation or upgrading to a new version.  
**Why it matters?** Proper installation ensures all features work correctly and avoids dependency conflicts.  
**Time to complete:** 5-10 minutes  
**Prerequisites:** Python 3.10 or 3.11 installed; pip package manager; terminal/command-line access

---

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
