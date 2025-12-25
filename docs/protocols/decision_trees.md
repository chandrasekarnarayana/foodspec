# Protocols: Decision Trees for Preprocessing and Model Choice

Use these flowcharts to choose preprocessing and models based on spectral issues and data characteristics.

## Preprocessing chooser
```mermaid
flowchart LR
  A[Raw spectra] --> B{Baseline drift/fluorescence?}
  B -->|Yes| C[Apply baseline correction (ALS or rubberband)]
  B -->|No| D{High noise?}
  D -->|Yes| E[Apply Savitzky–Golay smoothing; consider derivatives]
  D -->|No| F{Scatter/intensity differences?}
  F -->|Yes| G[Normalize (L2/area); consider SNV/MSC; crop edges]
  F -->|No| H[Proceed to feature extraction]
  C --> F
  E --> F
```

## Model chooser
```mermaid
flowchart LR
  A[Features ready] --> B{Task?}
  B -->|Classification| C{Data size / separability}
  C -->|Small / linear-ish| D[PLS-DA or Logistic/Linear SVM]
  C -->|Non-linear / more data| E[RBF SVM or Random Forest]
  B -->|Regression (mixtures)| F{Pure spectra available?}
  F -->|Yes| G[NNLS / PLS regression]
  F -->|No| H[MCR-ALS]
  B -->|QC / novelty| I[One-class SVM or IsolationForest]
```

## Notes
- Always inspect spectra before/after preprocessing.
- Align wavenumbers and metadata; use consistent pipelines for train/test.
- Validate models with appropriate CV and metrics; document choices.

## Further reading
- [Baseline correction](../../preprocessing/baseline_correction/)
- [Normalization & smoothing](../../preprocessing/normalization_smoothing/)
- [Classification & regression](../ml/classification_regression.md)
- [Mixture analysis workflow](../workflows/mixture_analysis.md)
