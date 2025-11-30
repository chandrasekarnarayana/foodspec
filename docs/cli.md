# Command Line Interface

The `foodspec` CLI provides quick, reproducible pipelines for preprocessing and oil authentication.

## Preprocess spectra

Run baseline correction, smoothing, normalization, and cropping on a folder of text spectra.

```bash
foodspec preprocess ./data/raw ./out/preprocessed.h5 \
  --metadata-csv ./data/metadata.csv \
  --modality raman \
  --min-wn 600 \
  --max-wn 1800
```

Arguments:
- `input_folder` (positional): folder with `.txt` spectra (two columns: wavenumber, intensity).
- `output_hdf5` (positional): path to write preprocessed HDF5.
- `--metadata-csv`: optional CSV with `sample_id` (matching filenames without extension) and extra fields.
- `--modality`: spectroscopy modality label.
- `--min-wn`, `--max-wn`: cropping bounds (cm^-1).

## Oil authentication

Run the edible oil authentication workflow on preprocessed spectra and produce an HTML report.

```bash
foodspec oil-auth ./out/preprocessed.h5 \
  --label-column oil_type \
  --output-report ./out/oil_auth_report.html
```

Arguments:
- `input_hdf5` (positional): HDF5 produced by preprocess step.
- `--label-column`: metadata column with class labels.
- `--output-report`: HTML report path.

Typical flow:
1. Prepare raw spectra (`raw/*.txt`) and `metadata.csv` with `sample_id` + labels.
2. Run `foodspec preprocess …` to produce `preprocessed.h5`.
3. Run `foodspec oil-auth …` to get `oil_auth_report.html`.
# Command Line Interface

The `foodspec` CLI provides quick, reproducible pipelines for preprocessing and oil authentication.

## Overview
- `foodspec preprocess`: baseline correction, smoothing, normalization, cropping; outputs HDF5.
- `foodspec oil-auth`: runs oil authentication workflow and writes an HTML report.

## foodspec preprocess

Run baseline correction, smoothing, normalization, and cropping on a folder of text spectra.

```bash
foodspec preprocess ./data/raw ./out/preprocessed.h5 \
  --metadata-csv ./data/metadata.csv \
  --modality raman \
  --min-wn 600 \
  --max-wn 1800
```

Arguments:
- `input_folder` (positional): folder with `.txt` spectra (two columns: wavenumber, intensity).
- `output_hdf5` (positional): path to write preprocessed HDF5.
- `--metadata-csv`: optional CSV with `sample_id` (matching filenames without extension) and extra fields.
- `--modality`: spectroscopy modality label.
- `--min-wn`, `--max-wn`: cropping bounds (cm^-1).

## foodspec oil-auth

Run the edible oil authentication workflow on preprocessed spectra and produce an HTML report.

```bash
foodspec oil-auth ./out/preprocessed.h5 \
  --label-column oil_type \
  --output-report ./out/oil_auth_report.html
```

Arguments:
- `input_hdf5` (positional): HDF5 produced by preprocess step.
- `--label-column`: metadata column with class labels.
- `--output-report`: HTML report path.

## Typical workflow
1) Prepare raw spectra (`raw/*.txt`) and `metadata.csv` with `sample_id` + labels.
2) Run `foodspec preprocess …` to produce `preprocessed.h5`.
3) Run `foodspec oil-auth …` to get `oil_auth_report.html`.
