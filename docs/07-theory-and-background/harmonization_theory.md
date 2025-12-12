# Theory – Harmonization & Calibration

Why harmonization matters for multi-instrument and multi-batch spectral studies:
- **Instrument drift**: wavenumber axes can shift over time; calibration curves correct drift and align spectra to a common grid.
- **Intensity scaling**: laser power and detector response vary; power/area normalization reduces cross-instrument intensity bias.
- **Residual variation**: diagnostics (pre/post alignment plots, residual metrics) quantify how well instruments agree after harmonization.
- **FAIR & reproducibility**: storing harmonization parameters and calibration metadata in HDF5/metadata ensures runs can be reproduced and audited.

For practical steps, see [hsi_and_harmonization.md](../05-advanced-topics/hsi_and_harmonization.md) and [cookbook_preprocessing.md](../03-cookbook/cookbook_preprocessing.md).
