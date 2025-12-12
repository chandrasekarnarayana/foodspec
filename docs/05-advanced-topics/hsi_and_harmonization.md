# Advanced Topic – HSI & Harmonization

This page covers hyperspectral imaging (HSI) workflows and cross-instrument harmonization—essential for spatial chemistry and multi-instrument studies.

## HSI pipeline
- **Data model**: `HyperspectralDataset` stores (x, y, wavenumber) cubes plus metadata.
- **Segmentation**: k-means, hierarchical, or NMF; configured via the `hsi_segment` step in protocols. Outputs label maps saved to the bundle (figures + arrays).
- **ROI extraction**: `hsi_roi_to_1d` step averages spectra per label/ROI and produces 1D spectra or peak/ratio tables used by RQ.

## Harmonization strategies
- **Wavenumber alignment**: align to a target grid via interpolation; calibration curves per instrument can correct drift.
- **Intensity/power normalization**: adjust for laser power/instrument response differences; uses metadata when available.
- **Diagnostics**: pre/post alignment plots and residual metrics can be saved to the bundle.
  - Bundles include harmonization diagnostics (e.g., mean overlay plots) when multiple instruments are harmonized; see `figures/harmonization_mean_overlay.png` and `metadata.json` → `harmonization`.

## Protocol representation
- HSI steps appear as `type: hsi_segment` and `type: hsi_roi_to_1d` in YAML protocols.
- Harmonization appears as `type: harmonize` with params like `align_to_common_grid`, `interp_method`, `calibration_curve`, `power_normalization`.

## Bundle outputs
- `figures/hsi_label_map*.png` (segmentation)
- `tables/roi_spectra.csv` (ROI averages) and downstream RQ tables
- Harmonization parameters logged in `metadata.json` and `index.json`

For step-by-step usage, see `02-tutorials/hsi_surface_mapping.md`. For theoretical background, see `07-theory-and-background/harmonization_theory.md`.
