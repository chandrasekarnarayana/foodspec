# FoodSpec Visualization & Reporting - Quick Start

## One-Liner: Auto-Generate Reports

```bash
# Run with automatic report generation
foodspec run --protocol examples/protocols/Oil_Auth.yaml \
    --input data/oils.csv \
    --output-dir runs/exp1 \
    --seed 42 \
    --report \
    --mode research
```

**Output**:
- `runs/exp1/report.html` — Full HTML report with all sections
- `runs/exp1/card.json` — Experiment card (JSON format)
- `runs/exp1/card.md` — Experiment card (Markdown format)

---

## Report Sections

### 1. Experiment Card (Top of Report)
- **Run ID & Timestamp**
- **Dataset Summary**: samples, modality, features
- **Model & Validation**: model name, CV scheme
- **Performance Snapshot**: F1, AUROC, ECE
- **Risk Assessment**: key risks identified
- **Confidence Level**: LOW / MEDIUM / HIGH with reasoning
- **Deployment Readiness**: NOT_READY / PILOT / READY

### 2. Performance Metrics
- Confusion matrix (classification)
- Per-class metrics (precision, recall, F1)
- Macro/weighted averages
- Fold-wise breakdown

### 3. Calibration & Uncertainty
- Reliability diagram (actual vs predicted probability)
- Expected Calibration Error (ECE)
- Coverage vs Efficiency curve (conformal prediction)
- Conformal set size distribution
- Abstention rate by class/batch

### 4. Interpretability
- Feature importance / coefficients
- Marker band plots (for spectral data)
- Feature stability maps (if available)

### 5. Drift & Stability
- Batch drift scores
- Temporal drift trends
- Stage-wise difference spectra
- Replicate similarity matrices

### 6. Provenance
- Protocol snapshot (expanded config)
- Protocol hash & data fingerprint
- Environment info (Python, OS)
- Run duration & timestamps

---

## Report Modes

### Research Mode (Default)
```bash
foodspec run --mode research  # Emphasizes analysis & interpretability
```
- All sections enabled
- Warnings OK (non-blocking)
- Best for: academic papers, exploration

### Regulatory Mode
```bash
foodspec run --mode regulatory  # Emphasizes QC & traceability
```
- Requires: manifest, metrics, QC, protocol snapshot, data fingerprint
- Warnings treated as errors
- Includes full audit trail
- Best for: GxP compliance, production validation

### Monitoring Mode
```bash
foodspec run --mode monitoring  # Emphasizes drift detection
```
- Drift dashboard prominent
- Comparison to baseline
- Minimal sections (metrics + uncertainty)
- Best for: ongoing system monitoring

---

## Paper-Ready Figures

### Export Figures for Publication

```python
from foodspec.reporting.export import export_paper_figures

# JOSS submission format
export_paper_figures(
    run_dir="runs/exp1",
    out_dir="submission/figures",
    preset="joss",  # or ieee, elsevier, nature
    formats=("png", "svg"),
)
```

**Output**:
```
submission/figures/
├── README.md
├── uncertainty/
│   ├── calibration.png (300 DPI)
│   └── calibration.svg (editable)
├── drift/
│   ├── batch_drift.png
│   └── batch_drift.svg
└── ...
```

### Figure Presets

| Preset | Best For | Size | DPI |
|--------|----------|------|-----|
| `joss` | arXiv, JOSS | 3.5 x 2.8 in | 300 |
| `ieee` | IEEE journals | 3.5 x 2.5 in | 300 |
| `elsevier` | Elsevier | 3.5 x 2.5 in | 300 |
| `nature` | Nature | Custom | 300 |

---

## Experiment Cards

### View in Browser
```bash
# Open card as Markdown
cat runs/exp1/card.md

# Or view JSON
cat runs/exp1/card.json | jq
```

### Example Card Output
```markdown
# Experiment Card

**Run ID**: abc1234  
**Timestamp**: 2025-01-26T10:30:00Z  
**Task**: classification | **Modality**: raman  

## Model

- **Type**: lightgbm
- **Validation**: stratified_kfold (5 folds)

## Performance Metrics

- **Macro F1**: 0.878
- **AUROC**: 0.925
- **ECE**: 0.0421
- **Coverage**: 92.0%
- **Abstention Rate**: 8.0%
- **Mean Set Size**: 1.15
- **Drift Score**: 0.095
- **QC Pass Rate**: 95.0%

## Summary

Excellent classification performance with good calibration. Low drift detected.
Model is ready for pilot deployment with standard monitoring.

## Risk Assessment

- Minor concern: High abstention rate (8.0%)

## Confidence & Readiness

**Confidence Level**: HIGH
No significant concerns identified

**Deployment Readiness**: PILOT
Standard monitoring recommended
```

---

## Programmatic API

### Load and Inspect Artifacts

```python
from pathlib import Path
from foodspec.reporting.base import ReportContext
from foodspec.reporting.cards import build_experiment_card

# Load all artifacts from run
context = ReportContext.load(Path("runs/exp1"))

# Access metrics
print(f"Metrics rows: {len(context.metrics)}")
print(f"First fold F1: {context.metrics[0]['macro_f1']}")

# Access trust outputs
print(f"ECE: {context.trust_outputs['calibration']['ece']:.4f}")
print(f"Coverage: {context.trust_outputs['conformal']['coverage_rate']:.1%}")

# Build card
card = build_experiment_card(context, mode="research")
print(f"Confidence: {card.confidence_level.value}")
print(f"Readiness: {card.deployment_readiness.value}")
```

### Generate Custom Report

```python
from foodspec.reporting.api import build_report_from_run

artifacts = build_report_from_run(
    run_dir="runs/exp1",
    mode="regulatory",
    pdf=True,  # Requires optional dependency
    title="Oil Authentication v1.0 GxP Report",
)

print("Generated artifacts:")
for name, path in artifacts.items():
    print(f"  {name}: {path}")
```

---

## Visualization Functions

### Individual Plots (Advanced)

```python
import numpy as np
from pathlib import Path
from foodspec.viz.comprehensive import (
    plot_raw_vs_processed_overlay,
    plot_coverage_efficiency_curve,
    plot_conformal_set_sizes,
)

# Raw vs processed overlay
fig = plot_raw_vs_processed_overlay(
    wavenumbers=np.linspace(500, 3000, 512),
    X_raw=X_raw,          # (n_samples, n_features)
    X_processed=X_processed,
    n_samples=5,
    seed=42,
    save_path=Path("raw_vs_processed.png"),
    dpi=300,
)

# Coverage efficiency curve
fig = plot_coverage_efficiency_curve(
    alpha_values=np.linspace(0, 0.5, 20),
    coverage=np.array([...]),
    efficiency=np.array([...]),
    target_alpha=0.1,
    save_path=Path("coverage_efficiency.png"),
)

# Conformal set sizes
fig = plot_conformal_set_sizes(
    set_sizes=np.array([1, 1, 2, 1, 1, ...]),
    labels=np.array([0, 1, 2, 0, 1, ...]),  # Optional: group by class
    save_path=Path("set_sizes.png"),
)
```

All visualization functions are **deterministic** (seed-controlled) and support:
- `seed` parameter for reproducibility
- `save_path` for automatic PNG/SVG export
- `dpi` for output resolution
- Return matplotlib Figure objects for customization

---

## Troubleshooting

### Report Not Generated?

1. **Check `--report` flag is enabled**
   ```bash
   foodspec run --protocol ... --report  # Default: True
   ```

2. **Check for errors in logs**
   ```bash
   foodspec run --verbose --report
   ```

3. **Verify artifacts exist**
   ```bash
   ls -la runs/exp1/
   # Should see: manifest.json, metrics.csv, predictions.csv, report.html
   ```

### Missing Sections?

Different report modes include different sections:

- **Research**: All sections (analysis-focused)
- **Regulatory**: Stricter validation (requires specific artifacts)
- **Monitoring**: Minimal sections (drift-focused)

If sections are missing, check:
1. Mode matches your data
2. Required artifacts are present
3. Try `--mode research` for most complete report

### Figure Quality Issues?

Use figure export for publication:

```python
from foodspec.reporting.export import export_paper_figures

export_paper_figures(
    run_dir="runs/exp1",
    preset="joss",    # Standardized styling
    formats=("png", "svg"),  # Vector + raster
)
```

This ensures consistent publication quality.

---

## Integration with Existing Workflows

### Protocol Runner
```python
from foodspec.protocol import ProtocolRunner

runner = ProtocolRunner.from_file("protocol.yaml")
result = runner.run([data])
runner.save_outputs(result, "runs/exp1")

# Reports auto-generated on CLI; programmatically:
from foodspec.reporting.api import build_report_from_run
build_report_from_run("runs/exp1")
```

### Experiment Orchestration
```python
from foodspec.experiment import Experiment

exp = Experiment.from_protocol("protocol_name")
result = exp.run(csv_path="data.csv", outdir="runs/exp1")

# Report already included in orchestration workflow
```

---

## FAQ

**Q: Can I customize the HTML template?**  
A: Not in current version. Use HTML report as starting point; custom styling can be added via CSS post-processing.

**Q: Are reports reproducible?**  
A: Yes! All plots are deterministic (seed-controlled). Same run + seed = identical report.

**Q: What if a figure doesn't generate?**  
A: Report generation continues gracefully. Missing artifacts noted in report ("Missing artifacts: drift/batch_drift.json").

**Q: Can I export to PDF?**  
A: Yes, if optional PDF dependencies installed. Add `pdf=True` to `build_report_from_run()` or use `--pdf` in CLI.

**Q: How large can reports get?**  
A: HTML reports are lightweight (~100 KB). Figure bundles depend on figure count (typically <10 MB for 100 figures).

---

## Next Steps

1. **Run your first report**: `foodspec run --report --mode research`
2. **Inspect the HTML**: Open `report.html` in browser
3. **Check the card**: View `card.md` for quick summary
4. **Export figures**: `export_paper_figures()` for publication
5. **Try different modes**: Experiment with `--mode regulatory` for compliance

---

**For full documentation**: See [VISUALIZATION_REPORTING_SUMMARY.md](VISUALIZATION_REPORTING_SUMMARY.md)
