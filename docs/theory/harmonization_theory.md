# Theory – Harmonization & Calibration

**Purpose:** Understand multi-instrument harmonization, calibration drift correction, and cross-batch transfer.  
**Audience:** Users working with multiple instruments or bridging data across time/batches.  
**Time to read:** 15–20 minutes.  
**Prerequisites:** Familiar with preprocessing steps and baseline correction.

---

Why harmonization matters for multi-instrument and multi-batch spectral studies:
- **Instrument drift**: wavenumber axes can shift over time; calibration curves correct drift and align spectra to a common grid.
- **Intensity scaling**: laser power and detector response vary; power/area normalization reduces cross-instrument intensity bias.
- **Residual variation**: diagnostics (pre/post alignment plots, residual metrics) quantify how well instruments agree after harmonization.
- **FAIR & reproducibility**: storing harmonization parameters and calibration metadata in HDF5/metadata ensures runs can be reproduced and audited.

For practical steps, see [hsi_and_harmonization.md](../theory/harmonization_theory.md) and [cookbook_preprocessing.md](../methods/preprocessing/normalization_smoothing.md).

---

## When Results Cannot Be Trusted

⚠️ **Red flags for harmonization validity:**

1. **Harmonization parameters fit on same data used for analysis (overfitting)**
   - Calibration curves optimized to training data may not generalize
   - Test data from same batch don't validate cross-instrument transfer
   - **Fix:** Use held-out data to fit calibration; validate transfer on independent instruments

2. **Wavenumber shift corrected but baseline residuals not checked (after alignment, baseline still tilted)**
   - Incomplete harmonization; residual differences confound analysis
   - Baseline correction parameters must also be harmonized
   - **Fix:** Visualize aligned spectra; check baseline and baseline-corrected spectra match post-harmonization

3. **Single reference standard used for calibration (all instruments calibrated to one oil, one day)**
   - Single point doesn't characterize full instrument behavior
   - Non-linearity or drift undetected
   - **Fix:** Use multiple reference standards (low, medium, high across relevant range)

4. **Harmonization parameters not validated on new instruments/times**
   - Parameters fit on Device A; deploying on Device B or future time without revalidation
   - Drift changes parameters; old calibration fails
   - **Fix:** Periodically revalidate harmonization; retrain if test-set performance degrades >5%

5. **Instrument differences masked but not removed (harmonization makes spectra similar, but underlying differences remain)**
   - Post-harmonization, residual variation still attributable to instrument
   - Analysis conclusions conflated with instrument artifacts
   - **Fix:** Document residual unexplained variance; test whether instrument explains batch effects

6. **Different preprocessing methods applied pre-harmonization (one instrument baseline-corrected, another not)**
   - Preprocessing fundamentally changes spectra; can't harmonize pre-processed to unprocessed
   - Calibration curves don't transfer
   - **Fix:** Apply identical preprocessing to all data before harmonization

7. **Harmonization assumes linear intensity correction, but non-linearity present**
   - Linear calibration (intensity = a*reference + b) may be oversimplified
   - Residuals show systematic patterns; correction incomplete
   - **Fix:** Check residuals; use higher-order or spline-based corrections if non-linearity detected

8. **Transfer learning (using parameters from one domain to harmonize another) without testing**
   - Example: oil-harmonization parameters applied to dairy without validation
   - Different matrices have different intensity/baseline characteristics
   - **Fix:** Retrain/validate parameters on target domain; don't assume transfer

---

## Next Steps

- [Data Formats & HDF5](../user-guide/data_formats_and_hdf5.md) — Store harmonized data efficiently.
- [Vendor I/O](../user-guide/vendor_io.md) — Handle multi-instrument data import.
- [Common Problems & Solutions](../troubleshooting/common_problems_and_solutions.md) — Troubleshoot harmonization failures.
