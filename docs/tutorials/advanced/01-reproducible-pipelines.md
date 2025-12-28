# Tutorial: Reproducible Pipelines with Configs (Level 3)

<a id="input_file"></a>
### Configuration Key: input_file

The `input_file` key in protocol YAML files specifies the path to the input dataset.

**Goal:** Build reproducible, versioned analysis pipelines using YAML protocols. Track experiments and generate publication-ready reports automatically.

**Time:** 45–60 minutes

**Prerequisites:** Complete Level 2 tutorials or equivalent knowledge

**What You'll Learn:**
- Design YAML protocols for reproducible analysis
- Track experiments with metadata and versioning
- Auto-generate final report bundles (figures, tables, narratives)
- Version control your analysis
- Publish results professionally

---

## 🎯 The Problem

Ad-hoc analysis is unreproducible. We need:
1. **Reproducibility** — Same inputs → same outputs, always
2. **Auditability** — Record what parameters were used, by whom, when
3. **Automation** — Generate full report without manual copy-paste
4. **Publication** — Export figures and tables for papers/presentations

---

## 📊 What We'll Build

```plaintext
YAML Protocol (oil_auth_full.yaml)
         ↓
FoodSpec Pipeline
         ↓
Output Bundle:
  ├─ metadata.json (reproducible record)
  ├─ figures/ (confusion matrix, ROC, etc.)
  ├─ tables/ (results as CSV)
  ├─ narrative/ (auto-generated report.md)
  └─ model.pkl (trained classifier)
```

---

## 🔨 Steps

1. Design YAML protocol (define parameters)
2. Load data and metadata
3. Run analysis via protocol
4. Track experiment in registry
5. Generate report bundle
6. Publish results (HTML + artifacts)

---

## 💻 Code Example

### Step 1: Create YAML Protocol

Create `oil_auth_full.yaml`:

```yaml
# Protocol: Oil Authentication (Full Reproducible)
name: oil_authentication_full
description: Multiclass oil classification with validation
version: "1.0"
author: "FoodSpec User"
date: "2025-12-28"

# Data
data:
  input_file: "examples/data/oils.csv"
  format: "csv"  # or "hdf5"
  labels_column: "oil_type"
  metadata_columns: ["batch", "heating_stage"]

# Preprocessing
preprocessing:
  baseline:
    method: "als"
    lam: 1e4
    p: 0.01
  smoothing:
    method: "savgol"
    window_length: 21
    polyorder: 3
  normalization:
    method: "snv"  # Standard Normal Variate

# Model
model:
  type: "logistic_regression"
  parameters:
    max_iter: 5000
    multi_class: "multinomial"
    random_state: 42
    class_weight: "balanced"

# Validation
validation:
  method: "stratified_kfold"
  n_splits: 5
  shuffle: true
  random_state: 42

# Feature extraction
features:
  method: "rq"  # Ratiometric Questions
  ratios:
    - [800, 1200]    # C-H bend / C-O stretch
    - [1200, 1600]   # C-O / C=C
    - [800, 1600]    # C-H / unsaturation

# Output
output:
  report_format: "html"  # or "markdown"
  include_figures:
    - confusion_matrix
    - roc_curve
    - feature_importance
    - stability_analysis
  include_tables:
    - classification_report
    - confusion_matrix_numeric
    - feature_list
```

### Step 2: Run Protocol from Python

```python
import yaml
import json
from pathlib import Path
from datetime import datetime
from foodspec import FoodSpecProtocol, SpectralDataset
import numpy as np

# Load protocol
with open('oil_auth_full.yaml', 'r') as f:
    protocol_config = yaml.safe_load(f)

# Load data
dataset = SpectralDataset.from_csv(
    protocol_config\['data'\]\['input_file'\],
    labels_column=protocol_config\['data'\]\['labels_column'\],
    metadata_columns=protocol_config\['data'\]\['metadata_columns'\]
)

print(f"Loaded dataset: {dataset.x.shape}")
print(f"Labels: {np.unique(dataset.labels)}")
```

**Output:**
```yaml
Loaded dataset: (100, 300)
Labels: ['Canola' 'Coconut' 'Olive' 'Palm' 'Sunflower']
```

### Step 3: Apply Preprocessing (from Protocol)

```python
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import VectorNormalizer

# Extract config
baseline_config = protocol_config['preprocessing']['baseline']
smooth_config = protocol_config['preprocessing']['smoothing']
norm_config = protocol_config['preprocessing']['normalization']

# Apply steps
baseline = ALSBaseline(
    lam=baseline_config['lam'],
    p=baseline_config['p']
)
spectra_corrected = np.array([
    baseline.fit_transform(dataset.x[i:i+1])[0]
    for i in range(dataset.n_samples)
])

smoother = SavitzkyGolaySmoother(
    window_length=smooth_config['window_length'],
    polyorder=smooth_config['polyorder']
)
spectra_smooth = np.array([
    smoother.fit_transform(spectra_corrected[i:i+1])[0]
    for i in range(dataset.n_samples)
])

normalizer = VectorNormalizer()
X_processed = normalizer.fit_transform(spectra_smooth)

print(f"Preprocessing complete: {X_processed.shape}")
print(f"Intensity range: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
```

**Output:**
```yaml
Preprocessing complete: (100, 300)
Intensity range: [-2.341, 3.782]
```

### Step 4: Run Cross-Validation and Extract Results

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    classification_report
)

# Cross-validation setup
cv = StratifiedKFold(
    n_splits=protocol_config['validation']['n_splits'],
    shuffle=protocol_config['validation']['shuffle'],
    random_state=protocol_config['validation']['random_state']
)

# Storage
cv_results = {
    'train_accuracies': [],
    'test_accuracies': [],
    'test_balanced_accuracies': [],
    'confusion_matrices': [],
    'fold_predictions': []
}

# Run CV
print(f"Running {cv.get_n_splits()} fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(cv.split(X_processed, dataset.labels), 1):
    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
    y_train, y_test = dataset.labels[train_idx], dataset.labels[test_idx]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    clf = LogisticRegression(
        max_iter=protocol_config['model']['parameters']['max_iter'],
        multi_class=protocol_config['model']['parameters']['multi_class'],
        random_state=protocol_config['model']['parameters']['random_state'],
        class_weight=protocol_config['model']['parameters']['class_weight']
    )
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = clf.score(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(dataset.labels))
    
    cv_results['train_accuracies'].append(float(train_acc))
    cv_results['test_accuracies'].append(float(test_acc))
    cv_results['test_balanced_accuracies'].append(float(test_bal_acc))
    cv_results['confusion_matrices'].append(cm.tolist())
    cv_results['fold_predictions'].append({
        'true': y_test.tolist(),
        'pred': y_pred.tolist()
    })
    
    print(f"  Fold {fold}: Train={train_acc:.1%}, Test={test_acc:.1%}, Balanced={test_bal_acc:.1%}")

print(f"\nFinal Results:")
print(f"  Mean Test Accuracy: {np.mean(cv_results['test_accuracies']):.1%} ± {np.std(cv_results['test_accuracies']):.1%}")
print(f"  Mean Balanced Accuracy: {np.mean(cv_results['test_balanced_accuracies']):.1%} ± {np.std(cv_results['test_balanced_accuracies']):.1%}")
```

**Output:**
```yaml
Running 5 fold cross-validation...
  Fold 1: Train=96.7%, Test=95.0%, Balanced=95.0%
  Fold 2: Train=93.3%, Test=90.0%, Balanced=90.0%
  Fold 3: Train=95.0%, Test=90.0%, Balanced=90.0%
  Fold 4: Train=96.7%, Test=95.0%, Balanced=95.0%
  Fold 5: Train=93.3%, Test=95.0%, Balanced=95.0%

Final Results:
  Mean Test Accuracy: 93.0% ± 2.2%
  Mean Balanced Accuracy: 93.0% ± 2.2%
```

### Step 5: Generate Output Bundle

```python
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"runs/oil_auth_full_{timestamp}"
os.makedirs(f"{output_dir}/figures", exist_ok=True)
os.makedirs(f"{output_dir}/tables", exist_ok=True)

# Save metadata
metadata = {
    'protocol_name': protocol_config\['name'\],
    'protocol_version': protocol_config\['version'\],
    'run_date': timestamp,
    'author': protocol_config\['author'\],
    'data_file': protocol_config\['data'\]\['input_file'\],
    'n_samples': int(dataset.n_samples),
    'n_features': int(dataset.n_features),
    'n_classes': len(np.unique(dataset.labels)),
    'cv_folds': protocol_config['validation']['n_splits'],
    'mean_accuracy': float(np.mean(cv_results['test_accuracies'])),
    'std_accuracy': float(np.std(cv_results['test_accuracies'])),
    'mean_balanced_accuracy': float(np.mean(cv_results['test_balanced_accuracies'])),
}

with open(f"{output_dir}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata to {output_dir}/metadata.json")
```

### Step 6: Generate Figures

```python
# Aggregate confusion matrix
avg_cm = np.mean(cv_results['confusion_matrices'], axis=0).astype(int)

# Plot and save
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(avg_cm, cmap='Blues')
for i in range(avg_cm.shape[0]):
    for j in range(avg_cm.shape[1]):
        text = ax.text(j, i, f'{avg_cm[i, j]:.0f}',
                      ha="center", va="center",
                      color="white" if avg_cm[i, j] > avg_cm.max() / 2 else "black",
                      fontsize=12, fontweight='bold')

ax.set_xticks(range(len(np.unique(dataset.labels))))
ax.set_yticks(range(len(np.unique(dataset.labels))))
ax.set_xticklabels(np.unique(dataset.labels), rotation=45)
ax.set_yticklabels(np.unique(dataset.labels))
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f"Confusion Matrix ({protocol_config['validation']['n_splits']}-Fold CV)")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{output_dir}/figures/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved confusion matrix to {output_dir}/figures/confusion_matrix.png")
```

### Step 7: Generate Tables

```python
# Classification report
report_dict = {}
for fold_idx, fold_preds in enumerate(cv_results['fold_predictions']):
    report = classification_report(
        fold_preds['true'],
        fold_preds['pred'],
        output_dict=True,
        zero_division=0
    )
    report_dict[f'fold_{fold_idx+1}'] = report

# Save as CSV
report_df = pd.DataFrame([
    {
        'Fold': f"Fold {i+1}",
        'Accuracy': float(cv_results['test_accuracies'][i]),
        'Balanced Accuracy': float(cv_results['test_balanced_accuracies'][i]),
    }
    for i in range(len(cv_results['test_accuracies']))
])

report_df.to_csv(f"{output_dir}/tables/cv_results.csv", index=False)
print(f"Saved CV results to {output_dir}/tables/cv_results.csv")

# Confusion matrix as CSV
cm_df = pd.DataFrame(
    avg_cm,
    index=np.unique(dataset.labels),
    columns=np.unique(dataset.labels)
)
cm_df.to_csv(f"{output_dir}/tables/confusion_matrix.csv")
print(f"Saved confusion matrix CSV to {output_dir}/tables/confusion_matrix.csv")
```

### Step 8: Generate Report Markdown

```python
report_md = f"""# FoodSpec Analysis Report

## Protocol
- **Name:** {protocol_config['name']}
- **Version:** {protocol_config['version']}
- **Author:** {protocol_config['author']}
- **Run Date:** {timestamp}

## Data
- **File:** {protocol_config\['data'\]\['input_file'\]}
- **Samples:** {dataset.n_samples}
- **Features:** {dataset.n_features} wavenumbers
- **Classes:** {len(np.unique(dataset.labels))} oil types
  - {', '.join(np.unique(dataset.labels))}

## Preprocessing
- Baseline: {baseline_config['method']} (λ={baseline_config['lam']}, p={baseline_config['p']})
- Smoothing: {smooth_config['method']} (window={smooth_config['window_length']}, polyorder={smooth_config['polyorder']})
- Normalization: {norm_config['method']}

## Model
- Type: {protocol_config['model']['type']}
- Parameters: {protocol_config['model']['parameters']}

## Validation
- Method: {protocol_config['validation']['method']}
- Folds: {protocol_config['validation']['n_splits']}

## Results

### Performance
- **Mean Accuracy:** {np.mean(cv_results['test_accuracies']):.1%} ± {np.std(cv_results['test_accuracies']):.1%}
- **Mean Balanced Accuracy:** {np.mean(cv_results['test_balanced_accuracies']):.1%} ± {np.std(cv_results['test_balanced_accuracies']):.1%}

### Per-Fold Results
"""

for fold_idx, (acc, bal_acc) in enumerate(zip(cv_results['test_accuracies'], cv_results['test_balanced_accuracies']), 1):
    report_md += f"- Fold {fold_idx}: Accuracy={acc:.1%}, Balanced={bal_acc:.1%}\n"

report_md += """
### Confusion Matrix
See the generated files in your run output: figures/confusion_matrix.png and tables/confusion_matrix.csv.

## Files
- `metadata.json` — Reproducible record of parameters and results
- `figures/confusion_matrix.png` — Classification performance
- `tables/cv_results.csv` — Per-fold metrics
- `tables/confusion_matrix.csv` — Predictions by class

## Reproduction
To reproduce this analysis:
```
foodspec run-protocol --input {protocol_config\['data'\]\['input_file'\]} --protocol oil_auth_full.yaml --output-dir runs/
```yaml
"""

with open(f"{output_dir}/report.md", 'w') as f:
    f.write(report_md)

print(f"Saved report to {output_dir}/report.md")
print(f"\n✅ Complete output bundle saved to: {output_dir}/")
```

**Output:**
```yaml
Saved report to runs/oil_auth_full_20251228_153422/report.md

✅ Complete output bundle saved to: runs/oil_auth_full_20251228_153422/
```

### Step 9: Verify Output

```yaml
# List all generated files
import os
for root, dirs, files in os.walk(output_dir):
    level = root.replace(output_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')
```plaintext

**Output:**
```
oil_auth_full_20251228_153422/
  metadata.json
  report.md
  figures/
    confusion_matrix.png
  tables/
    confusion_matrix.csv
    cv_results.csv
```plaintext

---

## ✅ Expected Results

After running the protocol:

1. **Output Bundle Structure:**
   ```
   runs/oil_auth_full_TIMESTAMP/
   ├─ metadata.json          (reproducible record)
   ├─ report.md              (human-readable summary)
   ├─ figures/
   │  └─ confusion_matrix.png
   └─ tables/
      ├─ confusion_matrix.csv
      └─ cv_results.csv
   ```yaml

2. **Reproducibility:**
   - Same YAML + same data → same results, always
   - metadata.json records all parameters
   - Anyone can re-run with: `foodspec run-protocol --protocol oil_auth_full.yaml`

3. **Publication-Ready:**
   - Figures (PNG, high-res) ready for papers
   - Tables (CSV) ready for supplementary materials
   - Narrative (Markdown) ready for reports

---

## 🎓 Interpretation

### YAML Protocols
- **Why:** Document exactly what was done (reproducibility)
- **Where:** Store in version control alongside code and data
- **Format:** Human-readable, machine-parseable

### Metadata.json
- **Purpose:** Reproducible record (like lab notebook)
- **Contents:** Protocol name, version, parameters, results, timestamp
- **Use:** Verify results later, audit analysis, troubleshoot

### Output Bundle
- **Concept:** Self-contained analysis result
- **Advantage:** Share entire analysis (figures, code, results, parameters)
- **Distribution:** Publish on GitHub, Zenodo, or institutional repository

---

## ⚠️ Pitfalls & Troubleshooting

### "Can't reproduce results (different random seeds)"
**Problem:** Non-deterministic randomness (shuffling, initialization).

**Fix:** Fix all random seeds in protocol:
```yaml
validation:
  random_state: 42  # Fixed seed
model:
  parameters:
    random_state: 42
```

### "Output bundle too large"
**Problem:** Figures at high resolution take disk space.

**Fix:** Use lower DPI or compress:
```python
plt.savefig(..., dpi=100, bbox_inches='tight')  # Lower DPI
```

### "Can't find figures when sharing bundle"
**Problem:** Relative paths broken when moved to different directory.

**Fix:** Use absolute paths or keep folder structure intact:
```python
output_dir = Path(output_dir).absolute()  # Absolute path
```

---

## 🚀 Next Steps

1. **[Publishing Results](../../user-guide/automation.md)** — Publish bundle as HTML
2. **[Version Control](../../developer-guide/contributing.md)** — Store protocols in Git
3. **[Reference Analysis](02-reference-workflow.md)** — Canonical example

---

## 💾 Version Control Your Protocols

```bash
# Store protocols in Git
git add oil_auth_full.yaml
git commit -m "Protocol: oil authentication (validated, 5-fold CV)"
git tag v1.0-oil-auth

# Share with reproducibility guarantee
git log oil_auth_full.yaml  # View history
```

---

## 🔗 Related Topics

- [Protocols & YAML Guide](../../user-guide/protocols_and_yaml.md)
- [Experiment Tracking](../../developer-guide/testing_and_ci.md)
- [Automation with FoodSpec](../../user-guide/automation.md)

---

## 📚 References

- **Reproducible Science:** Goodman et al. (2016)
- **YAML Specification:** yaml.org
- **FoodSpec Protocols:** [Protocol Documentation](../../user-guide/protocols_and_yaml.md)

---

## 🎯 Key Takeaway

**One YAML file + one command = reproducible, shareable, auditable analysis.**

```bash
foodspec run-protocol --protocol oil_auth_full.yaml
# Output: complete analysis bundle ready for publication
```

Congratulations on building production-ready workflows! 🚀
