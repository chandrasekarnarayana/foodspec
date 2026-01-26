# Feature Engineering

**Who needs this?** Food scientists and data scientists turning spectra into interpretable ML features.  
**What problem does this solve?** Converting spectra into peak, band, ratio, or embedding features with provenance.  
**When to use this?** When you must justify spectral markers (food safety, QA, research) and keep features reproducible.  
**Why it matters?** Interpretable features unblock safety reviews, auditing, and model updates without guesswork.  
**Time to complete:** 15 minutes  
**Prerequisites:** FoodSpec installed; CSV/HDF5 spectra with numeric wavenumber columns; protocol YAML defining peaks/bands.

## What?
- Peaks: height/area (optional width/centroid/symmetry) around user-defined centers; baselines per-peak.
- Bands: integrate intensity over ranges; optional local baseline per band.
- Ratios: numerator/denominator ratios between named peaks or bands.
- Embeddings: PCA or PLS/PLS-DA scores for compact chemometric summaries.
- Hybrid: concatenate peaks/bands with embeddings; scale via standard or robust scaling.

## Why?
- Spectroscopy-native features keep traceability to wavenumbers and chemistry assignments.
- Ratios and band integrals reduce instrument drift sensitivity while staying interpretable.
- PCA/PLS embeddings offer dimensionality reduction when full spectra are too wide.

## When?
- QA and authenticity checks where reviewers need band-level justifications.
- Model updates that must stay aligned with existing marker panels.
- Rapid prototyping with compact features before full-model training.

## How?

### 1) Define features in your protocol (YAML/JSON)

```yaml
name: Example_Features
expected_columns:
  label_col: label
features:
  peak_metrics: [height, area]
  band_metrics: [integral, mean]
  peak_window: 5.0
  peak_baseline: linear
  band_baseline: min
  peaks:
    - name: I_1742
      wavenumber: 1742
      window: 6
      baseline: linear
      assignment: "C=O stretch (example)"
    - name: I_1652
      wavenumber: 1652
      window: 6
      assignment: "C=C stretch (example)"
  bands:
    - name: band_1296_1310
      start: 1296
      end: 1310
      baseline: min
      assignment: "example lipid band"
  ratios:
    - name: 1742/1652
      numerator: I_1742
      denominator: I_1652
```

Oil band values such as 1296, 1434, 1652, 1742 cm^-1 are common **examples only** — verify against your matrix and instrument. If you omit assignments, FoodSpec records `unassigned` rather than guessing.

### 2) Extract features (CLI)

```bash
foodspec features extract \
  --csv examples/data/oil_synthetic.csv \
  --protocol examples/protocols/EdibleOil_Classification_v1.yaml \
  --type peaks \
  --outdir runs/features_peaks
```

### 3) Select a marker panel (stability selection)

```bash
foodspec features select \
  --run runs/features_peaks \
  --method stability \
  --k 8 \
  --outdir runs/features_panel
```

### 4) Use features in training

```bash
foodspec train \
  --csv examples/data/oil_synthetic.csv \
  --protocol examples/protocols/EdibleOil_Classification_v1.yaml \
  --features peaks \
  --scheme loso \
  --group heating_stage \
  --model logreg \
  --outdir runs/train_peaks
```

## Pitfalls & Limitations

### When Results Cannot Be Trusted

Results are unreliable when:
1. Peak windows miss the actual maxima (misaligned wavenumber axis or over-tight windows).
2. Band baselines are mis-set (e.g., using `min` on noisy spectra inflates integrals).
3. Stability selection is run with too few resamples (<10) leading to noisy frequencies.

**How to detect:**
- Plot extracted peaks/bands against spectra; check peak masks visually.
- Inspect `features/stability.csv` for high variance or flat frequencies.
- Validate that peak/band columns are non-null and within physical ranges.

**What to do:**
- Adjust `peak_window`/`band` ranges to match your instrument resolution.
- Switch baseline to `linear` or `median` if `min` over-corrects.
- Increase `--n-resamples` to 30+ for stable selection.

## Output artifacts
- `features/features.csv` and `features/feature_info.json` (assignments are preserved; unknowns are `unassigned`).
- `features/marker_panel.json` and `features/marker_panel.csv` with ranked panels.
- `features/stability.csv` for selection frequencies.
- Run metadata: `manifest.json`, `run_summary.json`, and `logs/run.log`.

## What's Next?
- See [API: Features](../api/features.md) for programmatic use.
- For chemometric theory, read [Chemometrics](../api/chemometrics.md).
- For workflow wiring, see [Workflows](../workflows/index.md).
