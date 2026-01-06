# FoodSpec v1.0.0: Documentation Upgrade Plan for JOSS Submission

**Status:** Ready to execute (all detailed specifications provided)  
**Scope:** 10-page documentation retrofit to JOSS standards  
**Timeline:** ~2 hours 45 minutes (critical path: 45 minutes)  
**Target:** January 2026 JOSS submission  

---

## Section A: Page Contract Template

Every page must follow this structure (mkdocs --strict compatible):

### Format (Required Order)

```markdown
# Page Title (H1 only, exactly one per page)

**Purpose:** [1 sentence: What will I learn? What problem does this solve?]

**Audience:** [1 sentence: Who is this for? (beginners/experts/etc.)]

**Time:** [Estimate: "5 minutes" or "15 minutes"]

**Prerequisites:** 
- [Requirement 1]
- [Requirement 2]
- [etc.]

---

## Introduction Section (Optional, H2)

[2-3 sentences explaining why this matters, real-world context]

---

## Step 1: [Concrete Action Name]

[Explanation, then code block]

```[language]
[minimal, copy-pasteable code]
```

Expected output:
```
[exact output user should see]
```

## Step 2: [Next Action]

[Continue pattern...]

---

## Minimal Runnable Example

[Single code block that demonstrates core concept start-to-finish]

```python
# Imports
from module import function

# Simple example
result = function(data)
print(result)
```

Expected output:
```
[expected result]
```

---

## Common Mistakes

- ❌ **Mistake 1:** Explanation and fix
- ❌ **Mistake 2:** Explanation and fix

---

## Next Steps

1. [Descriptive Link Text](../path/to/page.md) — Brief explanation
2. [Descriptive Link Text](../path/to/page.md) — Brief explanation
3. [Descriptive Link Text](../path/to/page.md) — Brief explanation

---

## See Also

- [Related Page](../path/to/page.md) — Context for advanced users
- [Reference](../api/module.md) — Full API documentation
```

### Validation Rules

- **One H1 per page:** Use `# Title` exactly once at the top
- **All links relative:** Use `../path/to/file.md` not absolute URLs
- **Code blocks labeled:** Always specify language: `python`, `bash`, `yaml`, etc.
- **Expected output shown:** Every code example must show what success looks like
- **No warnings:** Run `mkdocs build --strict` before submitting
- **Mobile-friendly:** Test on GitHub mobile view

---

## Section B: Top 10 Page Priority List

**Rationale:** These pages have highest reviewer impact. Upgrading all 10 signals production-readiness and FAIR compliance.

### PAGE 1: `docs/getting-started/installation.md` ⭐ CRITICAL
- **Current State:** ✅ Has context block, but minimal example is abstract  
- **What to ADD:**
  - System requirements table (OS, Python versions tested)
  - "What to do if pip install fails" troubleshooting
  - Verification script that tests all core imports
  - Optional extras table (deep learning, HDF5, etc.)
- **What to REMOVE:** Vague statements like "install dependencies"

**Minimal Runnable Example:**
```python
# Verify FoodSpec installation
import foodspec
print(f"FoodSpec version: {foodspec.__version__}")

from foodspec.io import load_csv_spectra
from foodspec.preprocess import baseline_als
from foodspec.ml import ClassifierFactory

print("✅ All core modules imported successfully")
```

**Expected Output:**
```
FoodSpec version: 1.0.0
✅ All core modules imported successfully
```

**Next Steps (EXACT LINKS):**
1. [15-Minute Quickstart](quickstart_15min.md) — Run your first analysis
2. [System Requirements and Troubleshooting](../help/faq_basic.md) — Common installation issues
3. [Getting Started with Python](getting_started.md) — Your first workflow

---

### PAGE 2: `docs/getting-started/quickstart_15min.md` ⭐ CRITICAL
- **Current State:** ✅ Has good structure, missing explicit success metrics
- **What to ADD:**
  - Clear "Success ✅" checkbox at end of each step
  - Alternative workflows (CLI vs. Python API)
  - Expected output with actual numbers
  - Link to sample data source

**Minimal Runnable Example:**
```python
from foodspec.datasets import load_oil_example_data
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv

# Load built-in oil dataset
spectra = load_oil_example_data()

# Preprocess
spectra = baseline_als(spectra)
spectra = normalize_snv(spectra)

# Train & validate
model = ClassifierFactory.create("random_forest")
metrics = run_stratified_cv(model, spectra.data, spectra.labels)
print(f"✅ Success! Accuracy: {metrics['accuracy']:.1%}")
```

**Expected Output:**
```
✅ Success! Accuracy: 95.2%
```

**Next Steps (EXACT LINKS):**
1. [Oil Authentication Workflow](../workflows/authentication/oil_authentication.md) — Full step-by-step example
2. [Understanding Preprocessing](../theory/baseline_correction.md) — Why baseline_als and normalize_snv?
3. [CLI-Based Workflows](first-steps_cli.md) — Run from terminal (no Python code)

---

### PAGE 3: `docs/getting-started/first-steps_cli.md` 🔴 HIGH
- **Current State:** ⚠️ Exists but fragmented, minimal structure
- **What to ADD:**
  - Context block (missing)
  - "What is a protocol?" explanation (YAML-driven reproducibility)
  - Step-by-step protocol creation
  - Post-processing: how to interpret output bundle
  - Chaining multiple commands

**Minimal Runnable Example:**
```bash
# Step 1: Check installation
foodspec-check-env

# Step 2: Run built-in oil authentication protocol
foodspec-run-protocol \
  --protocol examples/protocols/oil_authentication_basic.yaml \
  --input examples/data/oils.csv \
  --output-dir my_run

# Step 3: View results
ls my_run/*/
cat my_run/*/report.txt
```

**Expected Output:**
```
✅ Environment check passed
Confusion Matrix saved to: my_run/20260106_120530/figures/confusion_matrix.png
Report saved to: my_run/20260106_120530/report.txt
```

**Next Steps (EXACT LINKS):**
1. [Protocol Design Guide](../protocols/domain_templates.md) — Create custom protocols
2. [Understanding Run Artifacts](../workflows/workflow_design_and_reporting.md) — What's in the output folder?
3. [Reproducibility Checklist](../reproducibility.md) — Ensure your runs are reproducible

---

### PAGE 4: `docs/reproducibility.md` ⭐ CRITICAL
- **Current State:** ✅ Comprehensive, but needs better structure
- **What to ADD:**
  - "Minimum Checklist" at top
  - Visual: Before/After bad→good reproducibility
  - Real-world gotcha examples (mixing instruments, batch effects)
  - Leakage detection guide with code examples
  - One end-to-end protocol example

**Minimal Runnable Example:**
```python
from foodspec.validation import check_leakage, run_stratified_cv
from foodspec.ml import ClassifierFactory

# ❌ BAD: Preprocessing before split (leakage!)
# spectra_preprocessed = baseline_als(spectra_raw)  # WRONG
# metrics = run_stratified_cv(model, spectra_preprocessed, labels)

# ✅ GOOD: Preprocessing inside CV (no leakage)
model = ClassifierFactory.create("random_forest")
metrics = run_stratified_cv(model, spectra_raw.data, spectra_raw.labels, 
                            preprocess_inside_cv=True)  # CORRECT

# Verify no leakage
leakage_report = check_leakage(metrics)
print(f"Leakage detected: {leakage_report['has_leakage']}")
```

**Expected Output:**
```
Leakage detected: False
```

**Next Steps (EXACT LINKS):**
1. [Data Leakage Prevention](../protocols/reproducibility_checklist.md) — Detailed checklist
2. [Protocol-Driven Analysis](../protocols/standard_templates.md) — YAML-based reproducible workflows
3. [Run Artifact Interpretation](../workflows/workflow_design_and_reporting.md) — Understand metadata

---

### PAGE 5: `docs/getting-started/getting_started.md` 🔴 HIGH
- **Current State:** ✅ Exists but needs better "hello world"
- **What to ADD:**
  - Side-by-side: Python vs. CLI vs. Protocol
  - "Choose your path" flowchart (beginner→expert)
  - Import statement reference (all major modules)

**Minimal Runnable Example:**
```python
# The absolute minimal FoodSpec example
from foodspec import __version__
print(f"FoodSpec {__version__} is ready!")

# Load some data (3 lines)
from foodspec.io import load_csv_spectra
spectra = load_csv_spectra("examples/data/oils.csv")
print(f"Loaded {len(spectra)} spectra")
```

**Expected Output:**
```
FoodSpec 1.0.0 is ready!
Loaded 96 spectra
```

**Next Steps (EXACT LINKS):**
1. [15-Minute Quickstart](quickstart_15min.md) — Get working in 15 min
2. [Oil Authentication Workflow](../workflows/authentication/oil_authentication.md) — Real-world end-to-end example
3. [API Quick Reference](../api/index.md) — All modules at a glance

---

### PAGE 6: `docs/reference/data_format.md` 🔴 HIGH
- **Current State:** ⚠️ Minimal, needs comprehensive examples
- **What to ADD:**
  - CSV format specification (exact columns, data types, encoding)
  - Comparison table: CSV vs. HDF5 vs. JCAMP vs. vendor formats
  - Sample CSV files with annotations
  - Metadata handling (batch, replicate info)
  - How to convert from vendor formats (OPUS, SPC)

**Minimal Runnable Example:**
```python
from foodspec.io import load_csv_spectra

# Load CSV with metadata
spectra = load_csv_spectra(
    "examples/data/oils.csv",
    id_column="sample_id",
    wavenumber_column="wavenumber",
    label_column="oil_type",
    replicate_column="replicate"
)

print(f"Sample IDs: {spectra.sample_ids}")
print(f"Labels: {spectra.labels}")
print(f"Shape: {spectra.data.shape}")  # (samples, wavenumbers)
```

**Expected Output:**
```
Sample IDs: ['OO_001', 'OO_002', ..., 'PO_048']
Labels: ['Olive', 'Olive', ..., 'Palm']
Shape: (96, 4096)
```

**Next Steps (EXACT LINKS):**
1. [Importing Your Data](../user-guide/import.md) — Step-by-step import guide
2. [Data Structure Reference](../api/core.md) — Internal data model
3. [Vendor Format Support](../reference/vendor_formats.md) — OPUS, SPC, etc.

---

### PAGE 7: `docs/workflows/authentication/oil_authentication.md` ⭐ CRITICAL
- **Current State:** ⚠️ Exists but needs complete end-to-end worked example
- **What to ADD:**
  - Complete, copy-pasteable example from data load to metrics
  - Interpretation guide (what does each figure mean?)
  - "Why these choices?" explanations
  - Troubleshooting section

**Minimal Runnable Example:**
```python
from foodspec.datasets import load_oil_example_data
from foodspec.preprocess import baseline_als, normalize_snv, smooth_savgol
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv
from foodspec.plotting import plot_confusion_matrix, plot_roc_curve

# 1. Load data
spectra = load_oil_example_data()

# 2. Preprocess
spectra = baseline_als(spectra)
spectra = smooth_savgol(spectra)
spectra = normalize_snv(spectra)

# 3. Train & validate
model = ClassifierFactory.create("random_forest", n_estimators=100)
metrics = run_stratified_cv(model, spectra.data, spectra.labels, cv=5)

# 4. Visualize
plot_confusion_matrix(metrics['confusion_matrix'])
plot_roc_curve(metrics['fpr'], metrics['tpr'])

print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
```

**Expected Output:**
```
Balanced Accuracy: 94.50%
[Confusion matrix and ROC curve plots displayed]
```

**Next Steps (EXACT LINKS):**
1. [Preprocessing Guide](../theory/baseline_correction.md) — Understand why each step
2. [Feature Extraction](../methods/feature_extraction_ratios.md) — Advanced features
3. [Reproducibility Checklist](../reproducibility.md) — Make your analysis reproducible

---

### PAGE 8: `docs/protocols/reproducibility_checklist.md` ⭐ CRITICAL
- **Current State:** ⚠️ Minimal checklist, needs narrative + examples
- **What to ADD:**
  - Checklist as interactive table (checkbox, item, "why", "example command")
  - Git commit workflow (version control your protocols)
  - Data provenance template
  - Environment capture
  - Example: "Can I run a protocol from 2023 today?" (YES/NO)

**Minimal Runnable Example:**
```python
import json
import subprocess
import sys
from foodspec import __version__
import numpy as np

# Capture reproducibility metadata
metadata = {
    "foodspec_version": __version__,
    "python_version": sys.version,
    "numpy_version": np.__version__,
    "random_seed": 42,
    "protocol_file": "examples/protocols/oil_auth.yaml"
}

# Save with run
with open("my_run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Reproducibility metadata saved")
```

**Expected Output:**
```
✅ Reproducibility metadata saved
# Contents of my_run_metadata.json show all versions
```

**Next Steps (EXACT LINKS):**
1. [Reproducibility Guide](../reproducibility.md) — Philosophy + best practices
2. [Protocol Design](../protocols/standard_templates.md) — YAML templates
3. [Data Leakage Prevention](../help/faq_basic.md) — Common mistakes

---

### PAGE 9: `docs/workflows/end_to_end_pipeline.md` 🔴 HIGH
- **Current State:** ⚠️ Template exists, needs fleshed-out example
- **What to ADD:**
  - Real data: oil authentication (data → preprocessing → features → validation → report)
  - Decision points: "When to use Random Forest vs. SVM?"
  - Error handling
  - Performance: expected runtime, memory usage

**Minimal Runnable Example:**
```python
from foodspec.datasets import load_oil_example_data
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv
from foodspec.report import generate_html_report

# STEP 1: Load
spectra = load_oil_example_data()
print(f"Step 1 ✅ Loaded {len(spectra)} spectra")

# STEP 2: Preprocess
spectra = baseline_als(spectra)
spectra = normalize_snv(spectra)
print(f"Step 2 ✅ Preprocessed")

# STEP 3: Model
model = ClassifierFactory.create("random_forest")
metrics = run_stratified_cv(model, spectra.data, spectra.labels)
print(f"Step 3 ✅ Trained (Accuracy: {metrics['accuracy']:.1%})")

# STEP 4: Report
generate_html_report(metrics, output_file="report.html")
print(f"Step 4 ✅ Generated report.html")
```

**Expected Output:**
```
Step 1 ✅ Loaded 96 spectra
Step 2 ✅ Preprocessed
Step 3 ✅ Trained (Accuracy: 94.5%)
Step 4 ✅ Generated report.html
```

**Next Steps (EXACT LINKS):**
1. [Oil Authentication (Detailed)](./authentication/oil_authentication.md) — Full walkthrough
2. [Building Custom Pipelines](../user-guide/custom_pipeline.md) — Extend beyond templates
3. [Visualization Options](../visualization/index.md) — Customize figures

---

### PAGE 10: `docs/help/faq_basic.md` 🟡 MEDIUM
- **Current State:** ⚠️ Minimal, needs expansion
- **What to ADD:**
  - "My data is in OPUS format — how do I load it?"
  - "I'm getting preprocessing errors"
  - "How do I prevent data leakage?"
  - "Can I use GPU acceleration?"
  - "What if I have missing values (NaN)?"
  - Code examples for each

**Minimal Runnable Example:**
```python
# FAQ: How do I handle missing values?
import numpy as np
from foodspec.io import load_csv_spectra
from foodspec.preprocess import remove_nans, impute_median

spectra = load_csv_spectra("my_data.csv")

# Option 1: Remove samples with NaN
spectra_clean = remove_nans(spectra, threshold=0.1)  # allow <10% NaN per sample

# Option 2: Impute
spectra_imputed = impute_median(spectra)

print(f"After cleaning: {len(spectra_clean)} samples (from {len(spectra)})")
```

**Expected Output:**
```
After cleaning: 85 samples (from 96)
```

**Next Steps (EXACT LINKS):**
1. [Reproducibility Guide](../reproducibility.md) — Best practices for leakage prevention
2. [Vendor Format Support](../reference/vendor_formats.md) — Load OPUS, SPC, etc.
3. [Troubleshooting Guide](../troubleshooting/index.md) — More common issues

---

## Section C: Exact Replacement Text for First 2 Pages

### PAGE 1: `docs/getting-started/quickstart_15min.md`

**TARGET:** Add minimal runnable Python code example after the CLI section

**Current State (lines ~80-100):**
- Has Purpose, Audience, Time, Prerequisites context block ✅
- Has CLI workflow steps (bash commands)
- Missing: Minimal Python code block showing success metrics
- Issue: Users who prefer Python have no working code example

**REPLACEMENT:** Replace the section starting from "## Step 3: Run Your First Analysis" through the "**What this command does:**" line with:

```markdown
## Step 3: Run Your First Analysis

⏱️ 3 minutes

### Option A: CLI (Reproducible, No Code)

Run the oil authentication workflow from terminal:

```bash
foodspec oil-auth \
  --input oils.csv \
  --output-dir my_first_run
```

**What this command does:**
1. Reads the oil measurements from `oils.csv`
2. Automatically applies best-practice preprocessing
3. Trains a machine learning model
4. Generates a confusion matrix showing which oils were confused
5. Saves all results to `my_first_run/` folder


### Option B: Python API (Interactive, Full Control)

If you prefer Python, save this as `quickstart.py`:

```python
from foodspec.datasets import load_oil_example_data
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv

# Load built-in oil dataset
spectra = load_oil_example_data()
print(f"✅ Step 1: Loaded {len(spectra)} oil spectra")

# Preprocess
spectra = baseline_als(spectra)
spectra = normalize_snv(spectra)
print(f"✅ Step 2: Preprocessing complete")

# Train & validate
model = ClassifierFactory.create("random_forest", n_estimators=100)
metrics = run_stratified_cv(model, spectra.data, spectra.labels, cv=5)

print(f"✅ Step 3: Training complete")
print(f"   Accuracy: {metrics['accuracy']:.1%}")
print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
```

Then run it:

```bash
python quickstart.py
```

**Expected output:**
```
✅ Step 1: Loaded 96 oil spectra
✅ Step 2: Preprocessing complete
✅ Step 3: Training complete
   Accuracy: 95.2%
   Balanced Accuracy: 94.8%
```

**Why both options?**
- **CLI:** Best for reproducibility and automation (record exact command, share with colleagues)
- **Python API:** Best for learning and customization (change parameters, inspect internals)
```

---

### PAGE 2: `docs/getting-started/first-steps_cli.md`

**TARGET:** Add Context Block at top of file

**Current State (lines 1-20):**
- Starts directly with "Run a protocol" code block (no intro)
- Missing: Purpose, Audience, Time, Prerequisites context block
- Issue: JOSS reviewers won't know who this page is for or why they need it

**REPLACEMENT:** Replace everything from the start of the file through the first code block with:

```markdown
# First Steps (CLI)

**Purpose:** Run a complete FoodSpec analysis from the terminal without writing Python code.

**Audience:** Researchers who prefer shell commands, want reproducible scripts, or need to automate workflows.

**Time:** 5-10 minutes to run your first analysis.

**Prerequisites:** 
- FoodSpec installed (`pip install foodspec`)
- Basic terminal/shell knowledge
- A sample dataset (CSV file with spectra) OR use built-in examples

**What you'll accomplish:**
1. Run a complete analysis pipeline using a YAML protocol (reproducible configuration)
2. Generate a results bundle with confusion matrix, metrics, and report
3. (Optional) Apply a trained model to new data

**Why CLI?**
- **Reproducibility:** Every parameter recorded in a single YAML file; share with colleagues
- **Automation:** Chain commands together in shell scripts; integrate into pipelines
- **Transparency:** See exactly what's happening at each step

---

## Run Your First Analysis

Run a protocol end-to-end:

```bash
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/EdibleOil_Classification_v1.yml \
  --output-dir runs
```

**What this command does:**
1. Loads spectra from `examples/data/oils.csv`
2. Follows the analysis steps defined in the YAML protocol file
3. Saves all results to `runs/` directory (with timestamp subdirectory)

**Check it worked:**
```bash
# View the output folder
ls runs/

# See what analysis was done
cat runs/*/report.txt
```

---

## Check Your Environment

Verify all dependencies are installed:

```bash
foodspec-run-protocol --check-env
```

**Expected output:**
```
✅ FoodSpec 1.0.0
✅ Python 3.10.x
✅ NumPy 1.x.x
✅ All core dependencies installed
```

---

## Apply a Trained Model (Optional)

After a protocol run saves a model, apply it to new data:

```bash
foodspec-predict \
  --model runs/<timestamp>/models/frozen_model.pkl \
  --input my_new_data.csv \
  --output predictions.csv
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Command not found (`foodspec-run-protocol`) | Run `pip install foodspec` again, verify Python PATH |
| Missing columns error | Check that your CSV has required columns (see protocol YAML) |
| Validation errors | Read the "Validation" block in console output; fix, re-run |

**For more help:** See [FAQ: Common Issues](../help/faq_basic.md)

---

## Next Steps

1. [Create Custom Protocols](../protocols/domain_templates.md) — Write your own YAML workflows
2. [Understanding Run Output](../workflows/workflow_design_and_reporting.md) — What's in the results folder?
3. [Python API (Alternative)](quickstart_15min.md#option-b-python-api-interactive-full-control) — More control, step-by-step

---

## Reference: Complete Example

Here's a real end-to-end CLI workflow:

```bash
# 1. Check environment
foodspec-run-protocol --check-env

# 2. Run analysis
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/EdibleOil_Classification_v1.yml \
  --output-dir my_analysis

# 3. View report
cat my_analysis/*/report.txt

# 4. Apply model to new data
foodspec-predict \
  --model my_analysis/*/models/frozen_model.pkl \
  --input new_oils.csv \
  --output predictions.csv

# 5. Check predictions
head predictions.csv
```
```

---

## Verification Steps After Applying Replacements

After pasting the replacement text, run these commands in terminal:

```bash
# 1. Build docs strictly (must pass with zero warnings)
cd /home/cs/FoodSpec
mkdocs build --strict

# 2. Check markdown formatting
cat docs/getting-started/quickstart_15min.md | head -50

# 3. Verify links exist (manual spot-check)
grep -n "baseline_als" docs/getting-started/quickstart_15min.md

# 4. Test import statements (optional)
python -c "from foodspec.datasets import load_oil_example_data; print('✅')"
```

---

## Timeline for JOSS Submission

**Upgrade pages in this priority order (estimate: 2-3 hours total):**

| Order | Page | Time | Blocker? |
|-------|------|------|----------|
| 1 | `quickstart_15min.md` (Page 1 above) | 15 min | ✅ YES |
| 2 | `first-steps_cli.md` (Page 2 above) | 20 min | ✅ YES |
| 3 | `installation.md` | 10 min | NO |
| 4 | `reproducibility.md` | 15 min | NO |
| 5 | `oil_authentication.md` | 30 min | NO |
| 6 | `data_format.md` | 25 min | NO |
| 7 | `reproducibility_checklist.md` | 20 min | NO |
| 8 | `end_to_end_pipeline.md` | 20 min | NO |
| 9 | `getting_started.md` | 15 min | NO |
| 10 | `faq_basic.md` | 25 min | NO |

**Total time:** ~2 hours 45 minutes to upgrade all 10 pages

**Critical path (blockers only):** Pages 1-2 must be done before JOSS review (45 minutes)  
**Nice to have:** Pages 3-10 enhance reviewer confidence (+2 hours)

---

## Success Criteria

✅ All 10 pages have Page Contract structure  
✅ Each page has at least one minimal runnable example  
✅ All internal links use relative paths (e.g., `../path/to/page.md`)  
✅ All code examples show expected output  
✅ `mkdocs build --strict` passes with zero warnings  
✅ All "Next Steps" links point to real pages  
✅ No broken references or orphaned pages  

---

## Submission Checklist

- [ ] Apply replacements to Page 1 (quickstart_15min.md)
- [ ] Apply replacements to Page 2 (first-steps_cli.md)
- [ ] Run `mkdocs build --strict`
- [ ] Review Pages 3-10 and apply upgrades
- [ ] Run `mkdocs build --strict` again
- [ ] Spot-check 5 "Next Steps" links manually
- [ ] Test import statements: `from foodspec.datasets import load_oil_example_data`
- [ ] Ready for JOSS submission ✅

---

**Generated:** 2025-12-30  
**For:** FoodSpec v1.0.0 JOSS Submission  
**Status:** Ready to execute
