# User Guide – Protocols & YAML

Protocols are the heart of FoodSpec: YAML/JSON recipes that make analyses reproducible and auditable. This page defines the schema, shows examples, and explains how protocols are discovered by the CLI.

**Why it matters:** A clear protocol removes guesswork, enforces validation, and makes runs repeatable via the CLI. Versioning (`version`, `min_foodspec_version`) prevents incompatibilities.

## Protocol structure
Top-level keys:
- `name`: human-readable protocol name.
- `version`: protocol version string.
- `min_foodspec_version`: (optional) minimum FoodSpec library version required.
- `description`: short text describing the use case.
- `expected_columns`: required columns in the input data.
- `steps`: ordered list of step objects.

## Step types
Built-in step types include:
- `preprocess`: baseline/smoothing/normalization/peak extraction.
- `harmonize`: align wavenumbers, power normalization, instrument calibration.
- `qc_checks`: basic dataset QC (class counts, constants, NaNs).
- `hsi_segment`: segment an HSI cube (k-means/NMF/hierarchical).
- `hsi_roi_to_1d`: extract ROI spectra from HSI into 1D datasets/peak tables.
- `rq_analysis`: run RQ engine (stability, discrimination, trends, divergence, minimal panel, clustering).
- `output`: bundle reports, figures, tables, metadata.

Each step has `type:` plus parameters relevant to that step (see examples below).

## Example 1 – Oil discrimination (basic)
Inline comments show why certain params matter.
```yaml
name: oil_basic
version: 1.0
min_foodspec_version: 0.2.0
description: Basic edible oil discrimination with ratiometric features
expected_columns: [oil_type, heating_stage, replicate]
steps:
  - type: preprocess              # ALS baseline, Savitzky-Golay smoothing, ref peak norm
    params:
      baseline_method: als
      smoothing: savgol
      normalization_mode: ref_peak_2720
      peak_mode: max
  - type: qc_checks               # class counts, constants, NaNs
  - type: rq_analysis             # RQ metrics + minimal panel + clustering
    params:
      enable_minimal_panel: true
      enable_clustering: true
      validation_strategy: batch_aware
  - type: output
```

## Example 2 – Oil vs chips matrix comparison
```yaml
name: oil_vs_chips
version: 1.0
description: Compare matrix effects between oils and chips
expected_columns: [oil_type, matrix, heating_stage, replicate]
steps:
  - type: preprocess          # normalize to common reference peak
    params:
      normalization_mode: ref_peak_2720
  - type: harmonize           # align wavenumbers across instruments
    params:
      align_to_common_grid: true
  - type: rq_analysis         # oil-vs-chips divergence enabled
    params:
      enable_oil_vs_chips: true
      validation_strategy: stratified
  - type: output
```

## Example 3 – HSI segment → ROI → RQ
```yaml
name: hsi_segment_roi
version: 1.0
description: Segment an HSI cube, extract ROIs, run RQ on ROI spectra
expected_columns: [wavenumber_axis, intensity_cube]
steps:
  - type: hsi_segment         # k-means segmentation
    params:
      method: kmeans
      n_clusters: 4
  - type: hsi_roi_to_1d       # average spectra per label
    params:
      average: true
  - type: rq_analysis         # clustering on ROI spectra if desired
    params:
      enable_clustering: true
  - type: output
```

## Designing your own protocol (mini-workflow)
1. Define the **goal** (discrimination, trends, HSI segmentation, etc.).
2. List **expected columns** and metadata needed by your steps.
3. Choose **step order** (often preprocess → harmonize → qc_checks → analysis → output).
4. Set **validation strategy** (batch-aware, nested) and minimal panel/clustering options as needed.
5. Save as YAML/JSON; include `min_foodspec_version` to prevent incompatibilities.
6. Test via CLI; check validation and outputs.

## How protocols are discovered
- **CLI:** pass `--protocol path/to/protocol.yaml` or a known name if installed via plugin.  
- **Plugins:** can register protocols via entry points; see [registry_and_plugins.md](registry_and_plugins.md) and [writing_plugins.md](../06-developer-guide/writing_plugins.md).

For more on validation and harmonization options, see [validation_strategies.md](../05-advanced-topics/validation_strategies.md) and [hsi_and_harmonization.md](../05-advanced-topics/hsi_and_harmonization.md).
