# FAQ (Basic)

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Beginners with common questions; users encountering issues during first use.  
**What problem does this solve?** Quick answers to frequent questions without reading full documentation.  
**When to use this?** When stuck on installation, data format, or basic workflow questions.  
**Why it matters?** Saves time by addressing common confusion points upfront.  
**Time to complete:** 5-10 minutes.  
**Prerequisites:** None

---

## Installation & Setup

**How do I install FoodSpec?**

```bash
pip install foodspec
foodspec --version  # Verify
```

See [Installation Guide](installation.md) for detailed troubleshooting.

---

**How do I check my installation?**

```bash
foodspec about                    # System info
foodspec-run-protocol --check-env # Dependencies
```

---

## Data & Formats

**What data formats does FoodSpec accept?**

- **CSV** (recommended for beginners): wavenumber + intensity columns
- **HDF5** (efficient for large datasets)
- **Vendor formats**: SPC, JCAMP-DX (with optional vendor loader plugin)

See [Data Format Reference](../reference/data_format.md) for examples.

---

**What do I do if my data has missing values?**

Options:
1. **Remove rows:** If only a few spectra have NaN values, drop them before loading
2. **Impute:** Use mean/median of column
3. **Interpolate:** For sparse wavenumber gaps, use `scipy.interpolate.interp1d`

Example:
```python
import pandas as pd
df = pd.read_csv("mydata.csv")
df.dropna(inplace=True)  # Remove NaN rows
# Or:
df.fillna(df.mean(), inplace=True)  # Impute with column mean
```

---

**How do I load data from a vendor instrument (Thermo, Renishaw, etc.)?**

1. **Easiest:** Export to CSV from vendor software; load with `load_csv_spectra()`
2. **Advanced:** Use vendor-specific loader if available
   ```python
   from foodspec.io import load_spc_file  # Thermo SPC
   spectra = load_spc_file("mydata.spc")
   ```
3. **Not available?** Request in GitHub Issues or see [Writing Plugins](../developer-guide/writing_plugins.md)

---

## Workflows & Analysis

**Do I need ML experience to use FoodSpec?**

No. Protocols are predefined recipes. You pick one (e.g., oil discrimination), map your columns, and run. Defaults include sensible validation and minimal panels. See [First Steps (CLI)](first-steps_cli.md).

---

**What is a protocol, in simple terms?**

A YAML/JSON recipe defining preprocessing, harmonization, QC, HSI (optional), analysis, outputs, and validation strategy. It makes runs repeatable and shareable.

Example:
```yaml
name: oil_authentication
preprocessing:
  baseline: als
  normalize: snv
validation:
  cv_folds: 5
model:
  type: random_forest
  n_estimators: 100
```

See [Protocol Design](../workflows/domain_templates.md) for templates.

---

**Should I start with CLI or Python?**

**Start with CLI.** It's simpler, reproducible, and requires no coding:
```bash
foodspec-run-protocol \
  --input mydata.csv \
  --protocol myprotocol.yaml \
  --output-dir results
```

Once comfortable, switch to Python for customization. See [First Steps (CLI)](first-steps_cli.md).

---

**Can I use FoodSpec for matrices beyond oils?**

Yes. Protocols focus on oils/milk/flour, but any Raman/FTIR data can be processed. Adjust expected columns, peak definitions, and validation thresholds as needed.

See [Domain Templates](../workflows/domain_templates.md) for examples.

---

## Results & Outputs

**Where do my results go?**

Each run creates a timestamped folder with:
```
runs/<protocol>/<timestamp>/
├── report.txt/html      # Summary report
├── figures/             # Confusion matrix, ROC, PCA plots
├── tables/              # CSV with predictions, metrics
├── metadata.json        # All parameters used
└── models/              # Trained model (if saved)
```

The CLI prints the path. Example: `Results saved to: runs/oil_basic_demo/20250106_143022_run/`

---

**What do the metrics mean?**

- **Accuracy:** Fraction of correct predictions
- **Balanced Accuracy:** Accounts for class imbalance
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under receiver-operating-characteristic curve (0.5=random, 1.0=perfect)

See [Metrics Reference](../reference/metrics_reference.md) for full definitions.

---

**How do I apply a trained model to new data?**

```bash
foodspec-predict \
  --input new_samples.csv \
  --model runs/oil_demo/<timestamp>/models/frozen_model.pkl
```

The model applies the same preprocessing used during training.

---

## Troubleshooting

**I get "Missing columns" error**

**Cause:** Input CSV columns don't match protocol expectations.

**Solution:**
```bash
# Check protocol expectations
cat myprotocol.yaml | grep expected_columns

# Use example data to test first
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/oil_basic.yaml
```

---

## Extension & Customization

**Can I extend FoodSpec?**

Yes. Add:
- **Custom protocols** (YAML files)
- **Vendor loaders** (Python plugin)
- **Preprocessing methods** (register in plugin registry)

See [Writing Plugins](../developer-guide/writing_plugins.md).

---

## Getting More Help

**What if my question isn't here?**

- Check [Troubleshooting FAQ](../troubleshooting/troubleshooting_faq.md) for common issues
- Search [GitHub Issues](https://github.com/spectrometrist/FoodSpec/issues)
- Ask a question in GitHub Discussions

---

## Quick Links

- [Installation](installation.md) — Get FoodSpec running
- [15-Minute Quickstart](quickstart_15min.md) — Python example
- [First Steps (CLI)](first-steps_cli.md) — Command-line guide
- [Data Format](../reference/data_format.md) — Data schema
- [Protocols](../workflows/domain_templates.md) — YAML templates
