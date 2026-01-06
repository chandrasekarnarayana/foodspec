# First Steps (CLI)

<!-- CONTEXT BLOCK (mandatory) -->
**Purpose:** Run a complete FoodSpec analysis from the terminal without writing Python code.  
**Audience:** Researchers who prefer shell commands, want reproducible scripts, or need to automate workflows.  
**Time:** 5-10 minutes to run your first analysis.  
**Prerequisites:** FoodSpec installed (`pip install foodspec`); basic terminal/shell knowledge; sample dataset (CSV) or use built-in examples

---

## What You'll Accomplish

1. **Run a complete analysis** using a YAML protocol (reproducible configuration)
2. **Generate a results bundle** with confusion matrix, metrics, and report
3. **(Optional) Apply a trained model** to new data

---

## Why Use CLI?

- **Reproducibility:** Every parameter recorded in a single YAML file; share with colleagues
- **Automation:** Chain commands together in shell scripts; integrate into pipelines
- **Transparency:** See exactly what's happening at each step
- **No coding:** Just run commands; no Python required

---

## Run Your First Analysis

Complete example (oil discrimination):

```bash
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/oil_basic.yaml \
  --output-dir runs/oil_basic_demo
```

**Expected output:**
```
✓ Data validation passed
✓ Preprocessing complete
✓ Training/validation complete
✓ Report generated
Results saved to: runs/oil_basic_demo/20250105_143022_run/
```

---

## Results Bundle Structure

The `foodspec-run-protocol` command creates this folder structure:

```plaintext
runs/oil_basic_demo/<timestamp>/
├── report.txt                 # Summary report (plain text)
├── report.html                # Interactive HTML report
├── metadata.json              # All parameters used
├── index.json                 # Results summary
├── run.log                    # Complete execution log
├── figures/                   # Plots (confusion matrix, ROC, etc.)
├── tables/                    # CSV tables with detailed results
└── models/                    # Trained model (if saved)
    └── frozen_model.pkl
```

---

## Publish Results as Document

Create a Methods-style text + figures:

```bash
foodspec-publish runs/oil_basic_demo/<timestamp> \
  --fig-limit 6
```

This creates `methods_text.md` and key figures suitable for a paper or report.

For all figures (supplementary):
```bash
foodspec-publish runs/oil_basic_demo/<timestamp> \
  --include-all-figures
```

---

## Apply a Frozen Model (Optional)

If the run saved a trained model, apply it to new data:

```bash
foodspec-predict \
  --input new_samples.csv \
  --model runs/oil_basic_demo/<timestamp>/models/frozen_model.pkl
```

This applies the same preprocessing and feature definitions used during training.

---

## Environment Check

If dependencies seem missing:

```bash
foodspec-run-protocol --check-env
```

This verifies Python version, installed packages, and FoodSpec version.

---

## Troubleshooting

### Problem: "Missing columns" error

**Cause:** Input CSV columns don't match protocol expectations.

**Solution:**
```bash
# Check what columns the protocol expects
cat examples/protocols/oil_basic.yaml | grep expected_columns

# Use the example CSV unchanged to test first
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/oil_basic.yaml \
  --output-dir runs/test
```

---

### Problem: Validation errors

**What happens:** The CLI prints a "Validation" block; it lists blocking errors and warnings.

**Solution:**
1. Fix blocking errors in your data or protocol
2. Re-run the command
3. Warnings are captured in `metadata.json` and `run.log` (safe to ignore in testing)

---

### Problem: "Permission denied" when saving results

**Cause:** Output directory is not writable.

**Solution:**
```bash
# Create output directory with write permissions
mkdir -p runs/oil_basic_demo
chmod u+w runs/oil_basic_demo

# Re-run
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/oil_basic.yaml \
  --output-dir runs/oil_basic_demo
```

---

## Next Steps

✓ Analysis ran successfully?

1. **Examine results** → Open `report.html` in a web browser
2. **Understand the metrics** → [Metrics Reference](../reference/metrics_reference.md)
3. **Create custom protocol** → [Protocol Design](../workflows/domain_templates.md)
4. **Python API** → [15-Minute Quickstart](quickstart_15min.md)

---

## Questions this page answers

- How do I run FoodSpec from the command line?
- What parameters does a protocol need?
- What does the output folder contain?
- How do I apply a trained model to new data?
- Where are the validation errors printed?
