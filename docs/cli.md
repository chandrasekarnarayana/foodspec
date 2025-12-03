# CLI reference

Questions this page answers
- What commands does foodspec provide?
- What arguments/options do they take?
- What outputs should I expect?
- How do I troubleshoot common CLI issues?

## about
- Synopsis: `foodspec about`
- Purpose: print foodspec/Python versions, optional extras status.

## preprocess
- Synopsis: `foodspec preprocess INPUT_FOLDER OUTPUT_HDF5 [--metadata-csv PATH] [--modality MOD] [--min-wn VAL --max-wn VAL]`
- Required: `INPUT_FOLDER`, `OUTPUT_HDF5`
- Common options: modality (`raman`/`ftir`), wavenumber crop, metadata CSV.
- Example:
  ```bash
  foodspec preprocess data/raw_txt libraries/preproc.h5 --metadata-csv data/meta.csv --modality raman --min-wn 600 --max-wn 1800
  ```

## csv-to-library
- Synopsis: `foodspec csv-to-library CSV_PATH OUTPUT_HDF5 [--format wide|long] [--wavenumber-column COL] [--sample-id-column COL] [--intensity-column COL] [--label-column COL] [--modality MOD]`
- Purpose: convert wide/long CSV to HDF5 library.
- Example:
  ```bash
  foodspec csv-to-library data/oils.csv libraries/oils.h5 --format wide --wavenumber-column wavenumber --label-column oil_type --modality raman
  ```

## oil-auth
- Synopsis: `foodspec oil-auth INPUT_HDF5 [--label-column COL] [--classifier-name NAME] [--cv-splits N] [--output-dir DIR]`
- Purpose: oil authentication classification workflow.
- Example:
  ```bash
  foodspec oil-auth libraries/oils.h5 --label-column oil_type --classifier-name rf --cv-splits 5 --output-dir runs/oils
  ```
- Outputs: metrics JSON/CSV, confusion_matrix.png, report.md.

## heating
- Synopsis: `foodspec heating INPUT_HDF5 [--time-column COL] [--output-dir DIR]`
- Purpose: heating degradation ratios vs time with trend/ANOVA.
- Example:
  ```bash
  foodspec heating libraries/heating.h5 --time-column heating_time --output-dir runs/heating
  ```

## qc
- Synopsis: `foodspec qc INPUT_HDF5 [--model-type oneclass_svm|isolation_forest] [--label-column COL] [--output-dir DIR]`
- Purpose: novelty/quality-control scoring.
- Example:
  ```bash
  foodspec qc libraries/oils.h5 --model-type oneclass_svm --output-dir runs/qc
  ```

## domains
- Synopsis: `foodspec domains INPUT_HDF5 --type {dairy,meat,microbial} [--label-column COL] [--classifier-name NAME] [--cv-splits N] [--output-dir DIR]`
- Purpose: domain templates reusing oil-style workflow.

## mixture
- Synopsis: `foodspec mixture INPUT_HDF5 --pure-hdf5 PURE_PATH [--mode nnls|mcr_als] [--spectrum-index IDX] [--output-dir DIR]`
- Purpose: NNLS or MCR-ALS mixture analysis.

## hyperspectral
- Synopsis: `foodspec hyperspectral INPUT_HDF5 --height H --width W --target-wavenumber WN [--window VAL] [--output-dir DIR]`
- Purpose: build hyperspectral cube and plot intensity map.

## protocol-benchmarks
- Synopsis: `foodspec protocol-benchmarks --output-dir DIR`
- Purpose: run reference benchmarks on public datasets; emits metrics/report.

## reproduce-methodsx
- Synopsis: `foodspec reproduce-methodsx --output-dir DIR`
- Purpose: run MethodsX reproduction (classification + mixture + PCA) and save artifacts.

## model-info
- Synopsis: `foodspec model-info MODEL_BASEPATH`
- Purpose: print saved model metadata from model registry.

## Troubleshooting CLI
- Missing HDF5 or bad path: check file existence and permissions.
- Invalid CSV format: ensure correct `--format` and column names; wavenumber must be numeric/monotonic.
- Wrong label column: confirm metadata column name; use `--label-column`.
- Small class sizes: reduce `--cv-splits` if classes have very few samples.
- For detailed errors, inspect report folder (`summary.json`, `metrics.json`, `report.md`) and logs printed to stdout.

See also
- [csv_to_library.md](csv_to_library.md)
- [workflows/oil_authentication.md](workflows/oil_authentication.md)
- [keyword_index.md](keyword_index.md)
- [ftir_raman_preprocessing.md](ftir_raman_preprocessing.md)
