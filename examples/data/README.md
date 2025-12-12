# Example Data (Synthetic)

- `oil_synthetic.csv`: wide-format Raman-like intensities with metadata columns:
  - `oil_type` (class label), `matrix` (oil), `heating_stage` (0..N), `batch` (A/B)
  - Wavenumber columns as numeric strings (e.g., `1000`, `1010`, ...)

- `chips_synthetic.csv`: similar structure with `matrix` = chips.

- `hsi_synthetic.npz`: small synthetic hyperspectral cube with shape (y, x, wn).

Units:
- Wavenumber in cm^-1, intensities are arbitrary units (synthetic).

These are synthetic stand-ins for quick prototyping; replace with real data as needed.
