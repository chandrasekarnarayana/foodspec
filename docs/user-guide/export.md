# Reproducibility and Archive Export

Export analysis runs as reproducible, shareable archives with complete environment and protocol snapshots.

## Overview

The export system provides two main functions:

1. **`build_reproducibility_pack()`** - Create a complete reproducibility package with environment, protocols, and metadata
2. **`export_archive()`** - Generate stable, deterministic ZIP archives for distribution

## Key Features

- **Deterministic Output**: Archives are bit-identical across systems for reproducibility
- **Selective Inclusion**: Choose which components to include (dossier, figures, tables, bundle)
- **Environment Capture**: Automatic pip freeze for dependency tracking
- **Protocol Snapshots**: Both JSON and human-readable formats
- **Integrity Verification**: Built-in archive validation

## Usage

### Build Reproducibility Pack

Create a comprehensive package with all reproducibility information:

```python
from foodspec.reporting.export import build_reproducibility_pack

pack_dir = build_reproducibility_pack(
    run_dir="path/to/analysis_run",
    out_dir="path/to/output"
)
```

**Pack Contents:**

- `manifest.json` - Run metadata and parameters
- `protocol_snapshot.json` - Full protocol (structured format)
- `protocol_snapshot.txt` - Human-readable protocol
- `environment.txt` - Pip freeze snapshot
- `tables/` - Metrics, predictions, QC results
- `plots_index.txt` - Catalog of all plots
- `pack_metadata.json` - Pack generation information

### Export Archive

Create stable ZIP archives with selective component inclusion:

```python
from foodspec.reporting.export import export_archive

# Full archive
archive_path = export_archive(
    out_zip_path="analysis.zip",
    run_dir="path/to/run",
    include=("dossier", "figures", "tables", "bundle")
)

# Minimal archive (documentation + figures)
minimal_zip = export_archive(
    "analysis_minimal.zip",
    run_dir="path/to/run",
    include=("dossier", "figures")
)
```

**Include Options:**

- `"dossier"` - Scientific documentation (methods, results, appendix)
- `"figures"` - Plot files (PNG, PDF, SVG)
- `"tables"` - Data tables (metrics, predictions, QC)
- `"bundle"` - Complete analysis bundle

### Verify Archive

```python
from foodspec.reporting.export import verify_archive_integrity, get_archive_file_list

# Check integrity
is_valid = verify_archive_integrity("archive.zip")

# Get file list (deterministic order)
files = get_archive_file_list("archive.zip")
```

## Architecture

### ReproducibilityPackBuilder

Orchestrates creation of reproducibility packs with methods for each component:

- `_copy_manifest()` - Copy execution manifest
- `_export_protocol_snapshot()` - Export protocol to JSON and text
- `_create_environment_freeze()` - Capture pip dependencies
- `_copy_data_tables()` - Collect metrics and predictions
- `_create_plots_index()` - Generate plot catalog
- `_create_pack_metadata()` - Document pack contents

### ArchiveExporter

Creates stable ZIP archives with deterministic ordering:

- Sorts files by path for consistency
- Prevents duplicate entries
- Supports selective component inclusion
- Uses consistent compression level

## Deterministic Ordering

Archives maintain deterministic file ordering through:

1. **Sorted Path List**: All files sorted alphabetically by archive path
2. **Duplicate Prevention**: Identical arcnames skipped
3. **Consistent Compression**: Fixed compression level (6)
4. **Standard ZipInfo**: No variable metadata affecting hash

This ensures that:
- Same run → same hash (bitwise identical)
- Different systems → identical archives
- Auditable: archive order visible via `get_archive_file_list()`

## Examples

### Complete Workflow

```python
from foodspec.reporting.export import (
    build_reproducibility_pack,
    export_archive,
    verify_archive_integrity
)

# 1. Build reproducibility pack
pack_dir = build_reproducibility_pack(
    run_dir="oil_auth_run_001",
    out_dir="reproducibility_packs"
)
print(f"Pack created: {pack_dir}")

# 2. Export full archive
archive = export_archive(
    "oil_analysis_full.zip",
    "oil_auth_run_001"
)
print(f"Archive: {archive.name} ({archive.stat().st_size} bytes)")

# 3. Verify integrity
valid = verify_archive_integrity(
    archive,
    expected_files=["manifest.json", "protocol_snapshot.json"]
)
print(f"Archive valid: {valid}")
```

### Size Optimization

```python
# Create three archives at different detail levels
from pathlib import Path

run_dir = "analysis_run_001"

# Documentation only (smallest)
export_archive(
    "run_documentation.zip",
    run_dir,
    include=["dossier"]
)

# Documentation + visualization
export_archive(
    "run_with_figures.zip",
    run_dir,
    include=["dossier", "figures"]
)

# Complete archive (largest)
export_archive(
    "run_complete.zip",
    run_dir
)
```

### Batch Export

```python
from pathlib import Path

# Export multiple runs
runs = Path("runs").glob("run_*")
for run_dir in runs:
    archive_path = f"{run_dir.name}.zip"
    export_archive(archive_path, str(run_dir))
    print(f"✓ {archive_path}")
```

## Protocol Snapshot Format

The protocol snapshot is captured in both formats:

**JSON (protocol_snapshot.json):**
```json
{
  "name": "Oil Authentication v2.1",
  "version": "2.1",
  "steps": [
    {
      "name": "SNV Normalization",
      "type": "preprocessing",
      "parameters": {"method": "snv"}
    }
  ]
}
```

**Text (protocol_snapshot.txt):**
```
PROTOCOL SNAPSHOT
======================================================================

Name: Oil Authentication v2.1
Version: 2.1

PROCESSING STEPS
----------------------------------------------------------------------

Step 1: SNV Normalization
Type: preprocessing
Description: Apply Standard Normal Variate normalization
Parameters:
  - method: snv
```

## Environment Capture

The environment is captured via:

```
# environment.txt
# Python: 3.12.9 (...)
# Generated: 2024-01-15T10:30:00

numpy==1.24.3
scikit-learn==1.3.0
pandas==2.0.1
foodspec==1.0.0
...
```

This enables:
- Dependency tracking and versioning
- Reproduction in clean environments
- Availability assessment (pip check)

## Use Cases

### 1. Publication & Distribution

```python
# Create shareable archive for collaborators
export_archive(
    f"{publication_id}_data.zip",
    run_dir,
    include=["dossier", "tables"]
)
```

### 2. Regulatory Compliance

```python
# Complete archive with full audit trail
export_archive(
    f"regulatory_submission_{run_id}.zip",
    run_dir,
    include=["dossier", "figures", "tables", "bundle"]
)
```

### 3. Minimal Sharing

```python
# Send only documentation and plots
export_archive(
    f"{run_id}_presentation.zip",
    run_dir,
    include=["dossier", "figures"]
)
```

## API Reference

### Functions

#### `build_reproducibility_pack(run_dir, out_dir) → Path`

Build reproducibility pack from analysis run.

**Parameters:**
- `run_dir` (str|Path): Analysis run directory
- `out_dir` (str|Path): Output directory for pack

**Returns:** Path to pack directory

#### `export_archive(out_zip_path, run_dir, include=None) → Path`

Export analysis run to stable archive.

**Parameters:**
- `out_zip_path` (str|Path): Output zip file path
- `run_dir` (str|Path): Source run directory
- `include` (Sequence[str]): Components to include (default: all)

**Returns:** Path to created archive

#### `get_archive_file_list(zip_path) → list[str]`

Get deterministically sorted file list from archive.

**Parameters:**
- `zip_path` (str|Path): Path to archive

**Returns:** Sorted list of archive files

#### `verify_archive_integrity(zip_path, expected_files=None) → bool`

Verify archive integrity and optionally check expected files.

**Parameters:**
- `zip_path` (str|Path): Path to archive
- `expected_files` (Sequence[str]): Optional list of expected files

**Returns:** True if archive is valid

### Classes

#### `ReproducibilityPackBuilder`

Builder for reproducibility packs.

**Methods:**
- `build(run_dir, out_dir) → Path`: Build pack

#### `ArchiveExporter`

Exporter for stable archives.

**Methods:**
- `export(out_zip_path, run_dir, include=None) → Path`: Export archive

## Testing

All functionality is thoroughly tested:

```bash
pytest tests/reporting/test_export.py -v
```

**Test Coverage:**
- Pack creation and validation
- Archive generation and integrity
- Deterministic ordering
- Selective inclusion
- Error handling
- Integration workflows

## Performance Notes

- **Pack Creation**: ~1-2 seconds for typical runs
- **Archive Export**: Depends on archive size; typically <5 seconds
- **Archive Size**: Typically 1-10 MB for complete archives (compression reduces by ~70%)

## Common Issues

### Archive Verification Fails

Verify the archive hasn't been corrupted:
```python
if not verify_archive_integrity("archive.zip"):
    # Archive is corrupted or invalid
```

### Missing Expected Files

Check archive contents:
```python
files = get_archive_file_list("archive.zip")
if "manifest.json" not in files:
    # File not included
```

### Deterministic Ordering

To verify deterministic ordering:
```python
files1 = get_archive_file_list("archive1.zip")
files2 = get_archive_file_list("archive2.zip")
assert files1 == files2  # Same order
```
