## HSI and Harmonization Guide

- **HyperspectralDataset** (`spectral_dataset.py`): stores cubes (y, x, wn) with metadata, instrument info, segmentation labels, ROI masks; save/load to HDF5.
- **Segmentation**: `hsi_segment` protocol step uses k-means/hierarchical/NMF to produce label maps (saved as PNG/NPY).
- **ROI Extraction**: `hsi_roi_to_1d` converts labels/masks to averaged spectra, then peak tables; can invoke RQ on ROI data.
- **Harmonization**: simple wavenumber alignment (`align_wavenumbers`, `harmonize_datasets`) for multi-instrument grids; instrument metadata stored in HDF5 for reproducibility.
- Vendor IO tips: see `docs/vendor_io.md` for supported exports and limitations.
- **Outputs**: label maps saved under `figures/hsi/`, masks and label arrays under `hsi/`, ROI peak tables under `tables/` in each run bundle.
