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
HSI captures spatially resolved spectra; segmentation (k-means/NMF) reveals chemically distinct regions. ROI spectra can be analyzed with RQ to compare regions. See [hsi_and_harmonization.md](../05-advanced-topics/hsi_and_harmonization.md) and [harmonization_theory.md](../07-theory-and-background/harmonization_theory.md).

## GUI workflow
1) Launch `foodspec-gui`; open the **HSI** tab (or wizard step).  
2) Load `examples/data/hsi_cube.h5`; mean-intensity view should render.  
3) Select protocol: **“HSI segment + ROI → RQ”**.  
4) Configure segmentation (method, clusters) if exposed; defaults are fine.  
5) Run: segmentation produces label maps; ROIs are averaged and sent to RQ.  
6) Inspect:  
   - **HSI tab**: label map overlay, ROI summaries.  
   - RQ tabs: stability/discriminative outputs for ROI spectra.

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
![HSI label map](../assets/figures/hsi_label_map.png)
![ROI spectra](../assets/figures/roi_spectra.png)

## How to read the results
- **Label map**: visually inspect regions; confirm segmentation aligns with expected coatings/contaminants.  
- **ROI spectra**: compare averaged spectra per label; use discriminative/stability outputs to identify distinct regions.  
- If multiple cubes/instruments, ensure harmonization is enabled for wavenumber alignment/power normalization.

## Cross-links
- Getting started: [first-steps_gui.md](../01-getting-started/first-steps_gui.md)  
- Cookbook: [cookbook_preprocessing.md](../03-cookbook/cookbook_preprocessing.md)  
- Theory: [hsi_and_harmonization.md](../05-advanced-topics/hsi_and_harmonization.md)
