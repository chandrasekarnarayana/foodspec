# Example Catalog

This page catalogs all runnable examples in the FoodSpec repository. Each example demonstrates a complete workflow from data loading to analysis and reporting.

## Overview

FoodSpec includes **16 production-ready examples** covering the major food spectroscopy workflows. All examples are located in the `/examples/` directory and can be run directly with Python.

---

## Quick Start Examples

### 1. **Oil Authentication Quickstart** 
**File:** `examples/oil_authentication_quickstart.py`

Basic oil authentication workflow using FTIR/Raman spectroscopy.

**What it demonstrates:**
- Loading spectral data
- Preprocessing (baseline correction, normalization)
- PCA visualization
- Classification (PLS-DA, SVM)
- Model evaluation metrics

**Run:**
```bash
python examples/oil_authentication_quickstart.py
```

**Output:** Classification report, PCA plot, confusion matrix

---

### 2. **Heating Quality Quickstart**
**File:** `examples/heating_quality_quickstart.py`

Track oil degradation during thermal heating using spectral ratios.

**What it demonstrates:**
- Ratiometric question (RQ) engine
- Temporal tracking of degradation
- Quality scoring based on spectral changes
- Regression models for prediction

**Run:**
```bash
python examples/heating_quality_quickstart.py
```

**Output:** Degradation trajectory plot, quality scores, regression metrics

---

### 3. **Mixture Analysis Quickstart**
**File:** `examples/mixture_analysis_quickstart.py`

Quantify adulteration in food mixtures using spectroscopy.

**What it demonstrates:**
- Mixture modeling (NNLS, MCR-ALS)
- Component quantification
- Calibration curves
- Validation metrics (R², RMSE)

**Run:**
```bash
python examples/mixture_analysis_quickstart.py
```

**Output:** Mixture composition estimates, calibration plots

---

## Advanced Workflow Examples

### 4. **Hyperspectral Imaging Demo**
**File:** `examples/hyperspectral_demo.py`

Spatial mapping of chemical composition using HSI.

**What it demonstrates:**
- Loading hyperspectral datacubes
- ROI selection
- Spatial mapping of features
- False-color visualization

**Theory:** [Hyperspectral Mapping](../workflows/hyperspectral_mapping.md)

---

### 5. **Automated Analysis Script**
**File:** `examples/foodspec_auto_analysis_script.py`

Production-ready automated analysis pipeline.

**What it demonstrates:**
- Batch processing
- Protocol execution
- Automated reporting
- Registry integration

**Theory:** [Automated Analysis](../04-user-guide/automation.md)

---

### 6. **RQ Engine Demo**
**File:** `examples/foodspec_rq_demo.py`

Demonstrates the Ratiometric Question (RQ) engine for feature extraction.

**What it demonstrates:**
- Defining spectral ratios
- Batch ratio computation
- Interpretation and scoring
- Integration with classification

**Theory:** [RQ Engine Theory](../07-theory-and-background/rq_engine_theory.md)

---

## Data Governance & Reproducibility

### 7. **Governance Demo**
**File:** `examples/governance_demo.py`

Data governance, versioning, and audit trails.

**What it demonstrates:**
- Dataset versioning
- Metadata management
- Audit logging
- FAIR principles implementation

**Theory:** [Data Governance](../04-user-guide/data_governance.md)

---

## Domain-Specific Examples

### 8. **Oil Authentication (Full)**
**File:** `examples/oil_authentication_full.py`

Complete oil authentication workflow with validation.

**Features:**
- Multi-method comparison (PCA, PLS-DA, SVM, RF)
- Cross-validation
- External validation set
- Statistical significance testing

**Dataset:** Requires oil spectral data

---

### 9. **Heating Trajectory Analysis**
**File:** `examples/heating_trajectory_analysis.py`

Detailed heating degradation monitoring.

**Features:**
- Time-series analysis
- Degradation kinetics
- Quality thresholds
- Predictive modeling

**Dataset:** Requires thermal heating data

---

### 10. **Mixture Quantification**
**File:** `examples/mixture_quantification.py`

Advanced mixture analysis with MCR-ALS.

**Features:**
- Non-negative least squares (NNLS)
- Multivariate Curve Resolution (MCR-ALS)
- Concentration profiling
- Validation against known mixtures

---

## Quality Control Examples

### 11. **QC Workflow**
**File:** `examples/qc_workflow.py`

Quality control monitoring for production batches.

**Features:**
- Control chart generation
- Outlier detection (One-Class SVM, Isolation Forest)
- Drift monitoring
- Health scoring

**Theory:** [Batch QC](../workflows/batch_quality_control.md)

---

### 12. **Novelty Detection**
**File:** `examples/novelty_detection.py`

Detect anomalies and novel samples.

**Features:**
- One-Class SVM
- Autoencoder-based detection
- Confidence scoring
- Decision boundaries

---

## Calibration & Harmonization

### 13. **Calibration Transfer**
**File:** `examples/calibration_transfer.py`

Transfer models between instruments.

**Features:**
- Direct Standardization (DS)
- Piecewise Direct Standardization (PDS)
- Transfer validation
- Performance metrics

**Theory:** [Harmonization Theory](../07-theory-and-background/harmonization_theory.md)

---

### 14. **Multi-Instrument Harmonization**
**File:** `examples/multi_instrument_harmonization.py`

Harmonize data from multiple spectrometers.

**Features:**
- Batch effect correction
- SVA/ComBat methods
- Cross-instrument validation

---

## CLI & Automation

### 15. **CLI Workflow Example**
**File:** `examples/cli_workflow_example.sh`

Command-line interface demonstrations (bash script).

**Commands:**
```bash
# Run analysis
foodspec analyze --protocol oil_auth.yaml --input data.csv

# Run preprocessing
foodspec preprocess --baseline als --normalize snv data.csv

# Generate report
foodspec report --template oil_auth --output report.html
```

---

### 16. **Batch Processing**
**File:** `examples/batch_processing.py`

Process multiple datasets in parallel.

**Features:**
- Multi-file processing
- Parallel execution
- Progress tracking
- Consolidated reporting

---

## Example Data Requirements

Most examples can run with synthetic data, but for real-world results you'll need:

| Example | Data Type | Size | Source |
|---------|-----------|------|--------|
| Oil Authentication | FTIR/Raman spectra | 50-200 samples | [Public datasets](../datasets/dataset_design.md) |
| Heating Quality | Time-series spectra | 10-50 timepoints | Lab experiments |
| Mixture Analysis | Pure + mixture spectra | 20-100 samples | Controlled mixtures |
| Hyperspectral | HSI datacube | 100x100x500 | Imaging system |

---

## Running Examples

### Prerequisites
```bash
# Install FoodSpec with all dependencies
pip install foodspec[all]

# Or from source
cd FoodSpec
pip install -e .[all]
```

### Basic Usage
```bash
# Navigate to examples directory
cd examples

# Run any example
python oil_authentication_quickstart.py

# With custom data
python oil_authentication_quickstart.py --data /path/to/data.csv
```

### Troubleshooting

If examples fail:

1. **Check installation:**
   ```bash
   python -c "import foodspec; print(foodspec.__version__)"
   ```

2. **Verify dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check data paths:**
   - Examples expect data in `/data/` or synthetic generation
   - Adjust paths in script if needed

4. **See troubleshooting guide:** [Common Issues](../troubleshooting/common_problems_and_solutions.md)

---

## Contributing Examples

Have a workflow to share? See [Contributing Guide](../06-developer-guide/contributing.md) for:
- Example template
- Documentation requirements
- Testing standards
- Submission process

---

## Theory Connections

Each example connects to theoretical documentation:

- **Oil Authentication** → [Classification Theory](../07-theory-and-background/chemometrics_and_ml_basics.md)
- **Heating Quality** → [RQ Engine Theory](../07-theory-and-background/rq_engine_theory.md)
- **Mixture Analysis** → [Mixture Models](../ml/mixture_models.md)
- **QC Workflows** → [Batch QC](../workflows/batch_quality_control.md)
- **Harmonization** → [Harmonization Theory](../07-theory-and-background/harmonization_theory.md)

---

## See Also

- [Tutorials](../02-tutorials/oil_discrimination_basic.md) — Step-by-step learning paths
- [User Guide](../04-user-guide/automation.md) — Detailed feature documentation  
- [API Reference](../08-api/stats.md) — Function/class documentation
- [Protocols](../protocols/reference_protocol.md) — YAML-based analysis protocols
