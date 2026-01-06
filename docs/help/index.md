# Help & Support Center

Welcome to the FoodSpec Help Center. Find answers to common questions, troubleshoot errors, learn how to report issues, and discover best practices for reproducible spectroscopy.

---

## Quick Navigation

### [📋 Frequently Asked Questions (FAQ)](faq.md)
**5-10 minute read** | Conceptual questions and quick answers
- Installation and setup
- Data formats and I/O
- Preprocessing method selection
- Model choices and metrics
- Workflow decision-making

### [🔧 Troubleshooting Guide](troubleshooting.md)
**10-20 minute read** | Fix technical errors with step-by-step solutions
- Installation issues (pip, imports, Python version)
- Data validation and import errors
- NaN and missing value handling
- Baseline correction and preprocessing problems
- Classification and regression failures
- Diagnostic scripts and utilities

### [⚠️ Common Problems & Solutions](common_problems.md)
**20-30 minute read** | Comprehensive diagnosis across all workflow stages
- **Acquisition:** baseline drift, saturation, wavenumber misalignment, SNR
- **Metadata:** missing labels, class imbalance, mislabeled samples
- **Preprocessing:** over-smoothing, baseline issues, normalization
- **Machine Learning:** overfitting, data leakage, imbalanced performance
- **Statistics:** assumption violations, multiple comparisons
- **Visualization:** scales, labeling, overplotting
- **Workflow Design:** unclear questions, insufficient data

### [📝 Reporting & Reproducibility](reporting_and_reproducibility.md)
**10-15 minute read** | Document results for publication and peer review
- Core results and figures
- Supplementary material guidelines
- Describing methods reproducibly
- Follow-up validation tests
- FAIR data principles

### [📖 How to Cite](how_to_cite.md)
**2-5 minute read** | Citation formats for FoodSpec
- BibTeX and APA formats
- Acknowledging contributors
- Citing specific methods or datasets

---

## Browse by Task

### Getting Started
- [Installation Troubleshooting](troubleshooting.md#installation-issues)
- [Getting Started Guide](../getting-started/index.md)
- [Quickstart](../getting-started/quickstart_15min.md)

### Data & Metadata
- [Data Formats Guide](../reference/data_format.md)
- [Metadata Validation](troubleshooting.md#data-loading-issues)
- [Common Problems](common_problems.md) (Dataset & Metadata section - Missing labels, class imbalance)

### Preprocessing & Methods
- [Baseline Method Selection](faq.md#which-baseline-method-should-i-use)
- [Preprocessing Best Practices](../user-guide/index.md)
- [Common Problems](common_problems.md) - See Preprocessing section

### Machine Learning
- [Model Selection FAQ](faq.md)
- [Metrics Guide](../reference/metrics_reference.md)
- [Common Problems](common_problems.md) - See ML & DL sections

### Statistics & Reporting
- [Statistical Analysis Guide](../methods/index.md)
- [Reporting Guidelines](reporting_and_reproducibility.md)
- [Common Problems](common_problems.md) - See Statistics section

### Visualization
- [Plotting Best Practices](../user-guide/index.md)
- [Common Problems](common_problems.md) - See Visualization section

---

## Search by Error Message

### Installation Errors
| Error | Solution |
|-------|----------|
| `pip install foodspec` fails | [Troubleshooting → Installation](troubleshooting.md#installation-issues) |
| `ModuleNotFoundError: No module named 'foodspec'` | [Troubleshooting → Import Errors](troubleshooting.md#problem-import-errors-after-installation) |
| `Python version mismatch` | [Troubleshooting → Python Compatibility](troubleshooting.md#1-python-version-incompatibility) |

### Data Errors
| Error | Solution |
|-------|----------|
| Missing columns or labels | [Common Problems](common_problems.md) - Dataset & Metadata section |
| Invalid file format | [Common Problems](common_problems.md) - Operational Errors section |
| NaN or missing values | [Troubleshooting → Data Loading](troubleshooting.md#data-loading-issues) |

### Computation Errors
| Error | Solution |
|-------|----------|
| Baseline correction fails | [Common Problems](common_problems.md) - Preprocessing section |
| Overfitting / poor CV scores | [Common Problems](common_problems.md) - ML section |
| Data leakage detected | [Common Problems](common_problems.md) - ML section |
| Convergence or NaN during training | [Common Problems](common_problems.md) - Deep Learning section |

### Results Problems
| Error | Solution |
|-------|----------|
| Metrics don't match my goal | [Common Problems](common_problems.md) - Workflow Design section |
| Cannot reproduce results | [Reporting & Reproducibility](reporting_and_reproducibility.md) |
| Visualization looks wrong | [Common Problems](common_problems.md) - Visualization section |

---

## Diagnostic Tools

The FoodSpec library includes built-in diagnostics:

```python
from foodspec.validation import validate_spectrum_set
from foodspec.diagnostics import (
    estimate_snr,
    summarize_class_balance,
    detect_outliers,
    check_missing_metadata
)

# Check data validity
validate_spectrum_set(spectra)  # Validates structure, wavenumbers, data types

# Assess signal quality
snr = estimate_snr(spectrum)

# Analyze class distribution
summarize_class_balance(labels)  # Print class counts and imbalance ratio

# Find outliers
outliers = detect_outliers(X, method="pca_distance")

# Validate metadata
check_missing_metadata(df, required_cols=["sample_id", "label"])
```

---

## Still Need Help?

### Ask the Community
- [GitHub Discussions](https://github.com/chandrasekarnarayana/foodspec/discussions)
- [GitHub Issues](https://github.com/chandrasekarnarayana/foodspec/issues) (for bugs)

### Provide Helpful Bug Reports
When opening an issue, include:
1. FoodSpec version: `import foodspec; print(foodspec.__version__)`
2. Python version: `python --version`
3. Error traceback (full output)
4. Minimal reproducible example
5. System info (OS, conda/pip environment)

See [Reporting Guidelines](reporting_and_reproducibility.md) for details.

### Additional Resources
- [API Reference](../api/index.md) – Function documentation
- [User Guide](../user-guide/index.md) – Step-by-step tutorials
- [Theory](../theory/index.md) – Background on spectroscopy and chemometrics
- [Methods](../methods/index.md) – Detailed methodology documentation
- [Workflows](../workflows/index.md) – Pre-configured analysis workflows

---

## Page Structure

| Page | Length | Best For |
|------|--------|----------|
| [FAQ](faq.md) | 5-10 min | Quick answers to conceptual questions |
| [Troubleshooting](troubleshooting.md) | 10-20 min | Step-by-step error diagnosis |
| [Common Problems](common_problems.md) | 20-30 min | Deep-dive reference guide for all issues |
| [Reporting](reporting_and_reproducibility.md) | 10-15 min | Documentation and publication prep |
| [Citing](how_to_cite.md) | 2-5 min | Citation formats |

---

## Documentation Feedback

Found an error? Outdated information? Have a question that's not covered?
- [Open an issue](https://github.com/chandrasekarnarayana/foodspec/issues)
- [Start a discussion](https://github.com/chandrasekarnarayana/foodspec/discussions)
- [Submit a pull request](https://github.com/chandrasekarnarayana/foodspec/pulls)
6. **Report a bug** — Open a new GitHub Issue with a minimal reproducible example

---

**Start here:** Most questions are answered in the [FAQ](faq.md) or [Troubleshooting Guide](troubleshooting.md).
