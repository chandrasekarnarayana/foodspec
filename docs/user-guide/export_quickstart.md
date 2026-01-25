# Quick Start: Reproducibility & Archive Export

Create shareable, reproducible archives from your analysis runs in seconds.

## Installation

The export module is built into FoodSpec:

```python
from foodspec.reporting.export import (
    build_reproducibility_pack,
    export_archive,
)
```

## 30-Second Tutorial

### Export a Shareable Archive

```python
from foodspec.reporting.export import export_archive

# Create archive
archive = export_archive(
    out_zip_path="my_analysis.zip",
    run_dir="path/to/analysis_run"
)

print(f"Archive created: {archive}")
# Output: Archive created: /path/to/my_analysis.zip
```

Done! The archive is ready to share.

## Common Workflows

### 1. Export with Selective Components

```python
from foodspec.reporting.export import export_archive

# Documentation + Plots (smallest)
export_archive(
    "presentation.zip",
    "analysis_run",
    include=["dossier", "figures"]
)

# Complete archive (largest)
export_archive(
    "complete_analysis.zip",
    "analysis_run",
    include=["dossier", "figures", "tables", "bundle"]
)
```

### 2. Build Reproducibility Pack

```python
from foodspec.reporting.export import build_reproducibility_pack

pack = build_reproducibility_pack(
    run_dir="analysis_run",
    out_dir="reproducibility_packs"
)

print(f"Pack created at: {pack}")
```

**Pack includes:**
- Protocol (JSON + readable text)
- Environment freeze (pip snapshot)
- Execution manifest
- Metrics and predictions
- Plot index

### 3. Verify Archive

```python
from foodspec.reporting.export import verify_archive_integrity

# Check if valid
if verify_archive_integrity("analysis.zip"):
    print("✓ Archive is valid and uncorrupted")
else:
    print("✗ Archive is corrupted")

# Check for specific files
if verify_archive_integrity(
    "analysis.zip",
    expected_files=["manifest.json", "protocol_snapshot.json"]
):
    print("✓ All required files present")
```

### 4. Inspect Archive Contents

```python
from foodspec.reporting.export import get_archive_file_list

files = get_archive_file_list("analysis.zip")
for file in files:
    print(f"  {file}")
```

## Include Options

| Option | Contents | Use Case |
|--------|----------|----------|
| `"dossier"` | Methods, results, appendices | Documentation sharing |
| `"figures"` | PNG, PDF, SVG plots | Presentations |
| `"tables"` | Metrics, predictions, QC | Data analysis |
| `"bundle"` | Complete analysis package | Full reproducibility |

**Examples:**

```python
# Documentation only
export_archive("doc.zip", run_dir, include=["dossier"])

# Docs + visualization
export_archive("viz.zip", run_dir, include=["dossier", "figures"])

# Everything
export_archive("full.zip", run_dir)  # default includes all
```

## Use Cases

### 📄 Publication/Submission

```python
# Archive for journal submission
export_archive(
    f"manuscript_{submission_id}.zip",
    run_dir,
    include=["dossier", "figures", "tables"]
)
```

### 🔬 Regulatory Compliance

```python
# Complete archive for audit trail
export_archive(
    f"regulatory_{run_id}.zip",
    run_dir,
    include=["dossier", "figures", "tables", "bundle"]
)
```

### 📊 Presentation/Sharing

```python
# Minimal size for distribution
export_archive(
    f"presentation_{event}.zip",
    run_dir,
    include=["dossier", "figures"]
)
```

### 🔄 Reproducibility

```python
# Build full pack for reproduction
pack = build_reproducibility_pack(run_dir, "packs")

# Then export archive for sharing
export_archive(
    "reproducible_run.zip",
    run_dir
)
```

## Archive Contents

### Full Archive Structure

```
my_analysis.zip/
├── manifest.json                 # Run metadata
├── protocol_snapshot.json        # Protocol (structured)
├── metrics.json                  # Performance metrics
├── predictions.json              # Model predictions
├── dossier/
│   ├── methods.md               # Methods section
│   ├── results.md               # Results section
│   └── appendix.md              # Supporting info
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.pdf
│   └── feature_importance.png
└── bundle/
    └── processed_data.json      # Complete dataset
```

### Reproducibility Pack Structure

```
reproducibility_pack/
├── manifest.json                # Execution metadata
├── protocol_snapshot.json       # Full protocol (JSON)
├── protocol_snapshot.txt        # Human-readable protocol
├── environment.txt              # Pip freeze snapshot
├── pack_metadata.json           # Pack information
├── plots_index.txt              # Catalog of plots
└── tables/
    ├── metrics.json
    ├── predictions.json
    └── qc_results.json
```

## Batch Processing

Export multiple runs:

```python
from pathlib import Path
from foodspec.reporting.export import export_archive

# Export all runs
runs_dir = Path("analysis_runs")
for run_dir in runs_dir.glob("run_*"):
    archive_path = f"exports/{run_dir.name}.zip"
    export_archive(archive_path, str(run_dir))
    print(f"✓ {archive_path}")
```

## Deterministic Archives

Archives are reproducible - same run always produces identical archive:

```python
from foodspec.reporting.export import (
    export_archive,
    get_archive_file_list
)

# Create two archives from same run
archive1 = export_archive("run1.zip", "analysis_run")
archive2 = export_archive("run2.zip", "analysis_run")

# File lists are identical
files1 = get_archive_file_list(archive1)
files2 = get_archive_file_list(archive2)

assert files1 == files2  # ✓ Deterministic ordering
```

This means:
- ✅ Different systems produce same archive hash
- ✅ Reproducible audit trails
- ✅ Verifiable file ordering
- ✅ Shareable with confidence

## Troubleshooting

### Archive too large?

Use selective inclusion:
```python
# Reduce size by excluding tables and bundle
export_archive(
    "small.zip",
    run_dir,
    include=["dossier", "figures"]
)
```

### Archive corrupted?

Check integrity:
```python
from foodspec.reporting.export import verify_archive_integrity

if not verify_archive_integrity("archive.zip"):
    print("Archive is corrupted - re-export it")
```

### Want human-readable protocol?

The reproducibility pack includes both:
```python
from foodspec.reporting.export import build_reproducibility_pack

pack = build_reproducibility_pack(run_dir, "packs")

# Both files are created:
# pack/protocol_snapshot.json  (structured)
# pack/protocol_snapshot.txt   (readable)
```

## Full API

### Functions

```python
export_archive(out_zip_path, run_dir, include=None) -> Path
    Create stable archive with optional selective inclusion

build_reproducibility_pack(run_dir, out_dir) -> Path
    Build complete reproducibility pack

get_archive_file_list(zip_path) -> list[str]
    Get deterministically sorted file listing

verify_archive_integrity(zip_path, expected_files=None) -> bool
    Verify archive is valid and contains expected files
```

## Next Steps

- 📖 Read [full documentation](./export.md)
- 🔬 Run [demo script](../examples/export_demo.py)
- ✅ Check [test examples](../tests/reporting/test_export.py)
- 🚀 Use in your workflows

---

**One command creates a shareable, reproducible archive:**

```python
from foodspec.reporting.export import export_archive
export_archive("my_analysis.zip", "run_dir")
```
