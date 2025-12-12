# Cookbook: Troubleshooting

- Missing columns: check protocol `expected_columns` and your CSV headers.
- Parsing errors: export vendor data as CSV/HDF5; see docs/vendor_io.md.
- High feature-to-sample ratio: heed warnings; consider reducing features or adding samples.
# Cookbook – Troubleshooting

Common failure modes and how to fix them (if you see X, do Y).

## Protocol validation failed – missing column X
- **Looks like:** Validation block/error dialog naming a missing column.  
- **GUI:** Map the correct column in **Data & Mapping**, re-validate.  
- **CLI:** Add/rename the column in CSV or pick a matching protocol; re-run and check the validation block.

## Nested CV too slow
- **Looks like:** Long runtimes during validation/feature selection.  
- **Fix:** Reduce feature set (top-N by variance/importance), lower folds, or switch to batch-aware CV. Disable nested CV if not needed.

## HSI file too large / performance issues
- **Looks like:** Memory/time spikes during HSI segmentation.  
- **Fix:** Downsample/bin wavenumbers or spatially; crop ROI; if harmonization not needed, disable it. Run segmentation on a subset first.

## Harmonization/wavenumber mismatch
- **Looks like:** Errors about mismatched axes/lengths across datasets.  
- **Fix:** Enable `harmonize` (align_to_common_grid); ensure instruments share or map to a target grid; verify wavenumber columns are numeric.

## Vendor file parse errors
- **Looks like:** Error about missing header/block in vendor file.  
- **Fix:** Export as CSV/HDF5 if possible; use a vendor plugin. Follow the error hint (e.g., “looks like OPUS but lacks X; export ASCII/CSV”).

## Model/predict mismatch
- **Looks like:** Prediction fails or outputs NaN because features don’t match.  
- **Fix:** Use `foodspec-predict` with the frozen model from the run folder; ensure the same preprocessing config/feature definitions and matching target grid/validation strategy.
