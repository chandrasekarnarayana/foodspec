# Validation for Food Spectroscopy

FoodSpec prioritizes validation schemes that respect real-world structure (batch, stage, instrument).
Random CV can leak information in food datasets where samples are correlated by production or processing.

## Why LOSO/LOBO matter

LOSO (Leave-One-Stage-Out) simulates deployment to a new processing stage.
LOBO (Leave-One-Batch-Out) simulates deployment to a new batch or production lot.

```
LOSO example (stage groups):

Stage A:  [x x x x]  -> test
Stage B:  [x x x x]  -> train
Stage C:  [x x x x]  -> train

LOBO example (batch groups):

Batch 01: [x x x]    -> train
Batch 02: [x x x]    -> test
Batch 03: [x x x]    -> train
```

These schemes protect against optimistic scores caused by near-duplicate spectra across batches or stages.

## Default policy

- Nested CV is the default for model selection.
- LOBO/LOSO are required when batch or stage identifiers exist.
- Random CV is blocked unless explicitly enabled with `--unsafe-random-cv`.

## CLI usage

```bash
foodspec train --csv data/spectra.csv --protocol protocols/oil.yaml \
  --scheme loso --group stage --model lightgbm --outdir runs/oil_loso
```

```bash
foodspec evaluate --run runs/oil_loso --outdir runs/oil_loso_eval
```

## Artifacts

Training and evaluation runs emit:
- `models/best_model.joblib`
- `models/model_card.json`
- `validation/folds.json`
- `metrics/metrics.json`
- `metrics/metrics_by_group.json`

## Statistical soundness

FoodSpec supports:
- Bootstrap confidence intervals for accuracy, macro-F1, AUROC, and ECE.
- Optional ANOVA/MANOVA on features or embeddings (assumptions must be checked):
  - Normality within groups
  - Homogeneity of variance/covariance
  - Independence of samples
  - Multiple-comparison correction when testing many features

Use these diagnostics to report error bars and avoid over-claiming model performance.
