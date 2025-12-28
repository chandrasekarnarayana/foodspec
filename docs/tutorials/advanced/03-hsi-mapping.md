# Tutorial: HSI Surface Mapping

- Data: `examples/data/hsi_synthetic.npz`
- Notebook: `examples/notebooks/03_hsi_surface_mapping.ipynb`
- Protocol: HSI segmentation → ROI → RQ (see notebook example config).
- Visuals: Segmentation maps, ROI spectra, RQ summaries.
# Tutorial – HSI surface mapping

### What this tutorial covers
- **Problem:** Map surfaces/coatings/contaminants using hyperspectral imaging (HSI).  
- **Dataset:** `examples/data/hsi_cube.h5` (synthetic cube with wavenumber axis).  
- **Protocol:** `examples/protocols/hsi_segment_roi.yaml` (segmentation → ROI → RQ).

## Why it matters (theory)
HSI captures spatially resolved spectra; segmentation (k-means/NMF) reveals chemically distinct regions. ROI spectra can be analyzed with RQ to compare regions. See [hsi_and_harmonization.md](../../theory/harmonization_theory.md) and [harmonization_theory.md](../../theory/harmonization_theory.md).

<!-- GUI workflow removed; use CLI workflow below -->

## CLI workflow
```bash
foodspec-run-protocol \
  --input examples/data/hsi_cube.h5 \
  --protocol examples/protocols/hsi_segment_roi.yaml \
  --output-dir runs/hsi_demo

foodspec-publish runs/hsi_demo/<timestamp> --fig-limit 8
```
Check `figures/hsi_label_map*.png` and `tables/roi_spectra.csv` plus downstream RQ tables.

## Example figures (from run bundle)
![HSI label map](../../assets/figures/hsi_label_map.png)
![ROI spectra](../../assets/figures/roi_spectra.png)

## How to read the results
- **Label map**: visually inspect regions; confirm segmentation aligns with expected coatings/contaminants.  
- **ROI spectra**: compare averaged spectra per label; use discriminative/stability outputs to identify distinct regions.  
- If multiple cubes/instruments, ensure harmonization is enabled for wavenumber alignment/power normalization.

## Cross-links
- Cookbook: [cookbook_preprocessing.md](../../methods/preprocessing/normalization_smoothing.md)  
- Theory: [hsi_and_harmonization.md](../../theory/harmonization_theory.md)
