# Installation Guide

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Anyone wanting to install FoodSpec for food spectroscopy analysis.  
**What problem does this solve?** Setting up FoodSpec and its dependencies correctly.  
**When to use this?** First-time installation or upgrading to a new version.  
**Why it matters?** Proper installation ensures all features work correctly and avoids dependency conflicts.  
**Time to complete:** 5-10 minutes  
**Prerequisites:** Python 3.10 or 3.11 installed; pip package manager; terminal/command-line access

---

## System Requirements

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| **Python** | 3.10 | 3.11, 3.12 | Tested on 3.10–3.13 |
| **Memory** | 2 GB | 8+ GB | For large spectral libraries |
| **Disk** | 500 MB | 2+ GB | If using deep learning models |
| **OS** | Linux/macOS/Windows | Linux (recommended) | All major OSes supported |

---

## Quick Install (Core Package)

Standard FoodSpec with all essential features:

```bash
pip install foodspec
```

**Verify installation:**
```bash
foodspec --version
foodspec about
```

---

## Install with Deep Learning (Optional)

If you need CNN models (not required for basic use):

```bash
pip install "foodspec[deep]"
```

**Verify TensorFlow:**
```python
import tensorflow as tf
print(f"TensorFlow {tf.__version__} ready")
```

---

## Development Installation

If you want to contribute or modify code:

```bash
# Clone repository
git clone https://github.com/spectrometrist/FoodSpec.git
cd FoodSpec

# Install in editable mode with dev tools
pip install -e ".[dev]"

# Run tests to verify
pytest tests/ -v
```

---

## Verification Script

After installation, run this to verify all core modules work:

```python
# Verify FoodSpec installation
import foodspec
print(f"FoodSpec version: {foodspec.__version__}")

from foodspec.io import load_csv_spectra
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv

print("✅ All core modules imported successfully")
```

**Expected output:**
```
FoodSpec version: 1.0.0
✅ All core modules imported successfully
```

---

## Troubleshooting

### Problem: `pip install foodspec` fails

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Try again
pip install foodspec

# If still failing, check Python version
python --version  # Should be 3.10+
```

---

### Problem: `ImportError: No module named 'foodspec'`

**Solution:**
```bash
# Verify installation
pip list | grep foodspec

# Reinstall if needed
pip install --force-reinstall foodspec

# Check Python path
python -c "import sys; print(sys.executable)"
```

---

### Problem: Missing dependencies (numpy, scipy, etc.)

**Solution:**
```bash
# Install with all dependencies explicitly
pip install foodspec

# Or reinstall everything
pip uninstall foodspec -y && pip install foodspec
```

---

### Problem: "CUDA/TensorFlow errors" (with `[deep]`)

**Likely cause:** GPU or CUDA mismatch.

**Solution (CPU-only):**
```bash
pip uninstall tensorflow -y
pip install tensorflow-cpu
```

**Solution (GPU):**
- Ensure NVIDIA drivers: `nvidia-smi`
- Ensure CUDA 12.x compatible TensorFlow
- See [TensorFlow GPU setup](https://www.tensorflow.org/install/source)

---

### Problem: "Permission denied" on Linux/macOS

**Likely cause:** Installing to system Python.

**Solutions:**
```bash
# Install user-wide (recommended)
pip install --user foodspec

# Or use venv (best practice)
python -m venv foodspec_env
source foodspec_env/bin/activate  # Linux/macOS
pip install foodspec
```

---

### Problem: Import errors in Jupyter

**Likely cause:** Jupyter kernel using different Python than installation.

**Solution:**
```bash
# Create a dedicated Jupyter kernel
python -m ipykernel install --user --name foodspec
# Then select 'foodspec' kernel when opening notebook
```

---

## What Gets Installed?

### Core Dependencies (automatically installed)

- `numpy`, `scipy` — Numerical computing
- `pandas` — Data handling
- `scikit-learn` — Machine learning
- `pyyaml` — Protocol configs
- `matplotlib` — Plotting
- `h5py` — HDF5 files

### Optional (`[deep]`)

- `tensorflow` — Deep learning models

### Development (`[dev]`)

- `pytest` — Testing
- `sphinx` — Documentation
- `black` — Code formatting

---

## Using Conda (Alternative)

For isolated, reproducible environments:

```bash
# Create environment
conda create -n foodspec python=3.11 -y
conda activate foodspec

# Install FoodSpec
pip install foodspec

# Install optional deep learning
pip install "foodspec[deep]"
```

**Save environment:**
```bash
conda env export > foodspec_env.yml
```

**Recreate later:**
```bash
conda env create -f foodspec_env.yml
```

---

## Next Steps

✓ Installation verified?

1. **Python user:** → [15-Minute Quickstart](quickstart_15min.md)
2. **CLI user:** → [First Steps (CLI)](first-steps_cli.md)
3. **Real example:** → [Oil Authentication](../workflows/authentication/oil_authentication.md)

---

## Questions this page answers

- How do I install FoodSpec for my operating system?
- What are the hardware/software requirements?
- How do I verify the installation?
- How do I install optional dependencies (deep learning, dev tools)?
- What do I do if installation fails?
