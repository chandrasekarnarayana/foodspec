# User Guide

**Purpose:** Learn how to use FoodSpec for common tasks (loading data, running analyses, interpreting results).

**Audience:** Spectroscopy researchers and data scientists using FoodSpec in practice.

**Time:** 2–3 hours to read core sections; reference chapters as needed.

**Prerequisites:** Python basics; some familiarity with spectroscopy or food science.

---

## What You'll Learn

This guide covers practical FoodSpec workflows:
- **Loading data:** CSV, HDF5, vendor formats (OPUS/WiRE)
- **Preprocessing:** Baseline correction, smoothing, normalization
- **Analysis:** Classification, regression, mixture analysis
- **Quality control:** Validation, outlier detection, reproducibility
- **Customization:** Writing protocols, using plugins, extending FoodSpec

**For quick start:** See [15-minute quickstart](../getting-started/quickstart_15min.md)  
**For theory:** See [theory section](../theory/index.md)  
**For API details:** See [API reference](../api/core.md)

## Example: Typical Workflow

```python
from foodspec.io import load_csv
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.apps.oils import run_oil_authentication

# 1. Load (see: Data Formats & I/O)
ds = load_csv("oils.csv", wavenumber_col="wavenumber")

# 2. Preprocess (see: Preprocessing & Feature Extraction)
ds = baseline_als(ds, lam=1e6)
ds = normalize_snv(ds)

# 3. Analyze (see: Analysis Workflows)
results = run_oil_authentication(ds, label_column="oil_type")
print(f"Accuracy: {results.metrics['accuracy']:.2%}")

# 4. Validate (see: Validation & Reproducibility)
print(f"95% CI: [{results.metrics['accuracy_ci_lower']:.2%}, {results.metrics['accuracy_ci_upper']:.2%}]")
```

## Command-Line Interface

- **[CLI Overview](cli.md)** — Introduction to the command-line interface
- **[CLI Guide](cli.md)** — Detailed guide to all CLI commands
- **[CLI Help Reference](cli_help.md)** — Complete command reference

## Data Management

### Input/Output
- **[Data Formats & HDF5](data_formats_and_hdf5.md)** — Supported file formats and HDF5 libraries
- **[Vendor I/O](vendor_io.md)** — Import data from instrument vendors (Thermo, Bruker, etc.)
- **[CSV to Library](csv_to_library.md)** — Convert CSV files to FoodSpec libraries

### Libraries & Organization
- **[Libraries](libraries.md)** — Create and manage spectral libraries
- **[Library Search](library_search.md)** — Query and filter spectral data
- **[Data Governance](data_governance.md)** — Data provenance, versioning, and integrity

## Configuration & Automation

- **[Protocols & YAML](protocols_and_yaml.md)** — Define reproducible analysis workflows
- **[Protocol Profiles](protocol_profiles.md)** — Reusable configuration templates
- **[Automation](automation.md)** — Batch processing and scripting
- **[Config & Logging](config_logging.md)** — Configure FoodSpec behavior
- **[Logging](config_logging.md)** — Control output verbosity and log files

## Extensibility

- **[Registry & Plugins](registry_and_plugins.md)** — Extend FoodSpec with custom methods

## Related Sections

- **[Getting Started](../getting-started/installation.md)** — Installation and quickstarts
- **[Methods](../methods/index.md)** — Validation and chemometrics guides
- **[API Reference](../api/index.md)** — Python API documentation

---

**Use this when:** You need comprehensive documentation for a specific feature or workflow.
