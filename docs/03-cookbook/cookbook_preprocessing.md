# Cookbook: Preprocessing

- Baseline choices (ALS, rubberband, polynomial).
- Smoothing (Savitzky–Golay).
- Normalization (reference, vector, area, max).
- Harmonization options; see docs/05-advanced-topics/hsi_and_harmonization.md.
# Cookbook – Preprocessing recipes

Short, practical “How do I…?” tasks. Each recipe includes a CLI and Python path.

## Normalize spectra by a reference band
**Problem:** Normalize intensities to a reference band (e.g., 2720 cm⁻¹) to stabilize ratios.

- **CLI way:** Use a protocol with `normalization_mode: ref_peak_2720` (or override via flags if enabled).  
  ```bash
  foodspec-run-protocol --input my.csv --protocol examples/protocols/oil_basic.yaml
  ```
- **Python way:**  
  ```python
  from foodspec.preprocessing_pipeline import run_full_preprocessing, PreprocessingConfig
  cfg = PreprocessingConfig(normalization_mode="ref_peak_2720")
  df_norm = run_full_preprocessing(df_raw, cfg)
  ```
- **More:** [cookbook_rq_questions.md](cookbook_rq_questions.md), [rq_engine_theory.md](../07-theory-and-background/rq_engine_theory.md)

## Remove cosmic spikes
**Problem:** Spikes/cosmic rays in spectra.

- **CLI way:** Use a protocol with `remove_spikes: true` in preprocess.  
  ```bash
  foodspec-run-protocol --input my.csv --protocol examples/protocols/oil_basic.yaml
  ```
- **Python way:**  
  ```python
  cfg = PreprocessingConfig(remove_spikes=True)
  df_clean = run_full_preprocessing(df_raw, cfg)
  ```
- **More:** [ftir_raman_preprocessing.md](../ftir_raman_preprocessing.md)

## Align spectra from two instruments
**Problem:** Need a common wavenumber grid across instruments/batches.

- **CLI way:** Use a protocol with `harmonize` step (`align_to_common_grid`, `interp_method`).  
  ```bash
  foodspec-run-protocol --input instA.csv --input instB.csv --protocol examples/protocols/oil_vs_chips.yaml
  ```
- **Python way:**  
  ```python
  from foodspec.harmonization import align_wavenumbers
  aligned = align_wavenumbers([ds1, ds2], target_grid=None, method="interp")
  ```
- **More:** [hsi_and_harmonization.md](../05-advanced-topics/hsi_and_harmonization.md), [harmonization_theory.md](../07-theory-and-background/harmonization_theory.md)
