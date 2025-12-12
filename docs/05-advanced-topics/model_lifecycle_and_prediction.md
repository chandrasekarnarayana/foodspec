# Advanced Topic – Model Lifecycle & Prediction

FoodSpec supports training, freezing, and reusing models with embedded preprocessing and feature definitions.

## Key classes
- **TrainablePipeline**: runs preprocessing + feature construction + model training (e.g., logistic regression, random forest) with validation.
- **FrozenModel**: serialized package containing preprocessing config, feature definitions, model weights/coefficients, normalization steps, and version tags (protocol/library).

## Export and reuse
- During protocol runs, models can be frozen and saved under `models/` in the bundle. `metadata.json` and `index.json` record the model path and features used; the registry can log provenance.
- `foodspec-predict` (CLI) and GUI “Apply existing model” load a `FrozenModel` and apply the same preprocessing/feature extraction to new data.

## Use cases
- Routine QA: reuse a validated model on new batches/instruments with harmonization applied.
- Minimal marker panels: deploy compact feature sets with known accuracy.

## Validation interplay
- Validation strategy (batch-aware/nested) during training influences reported metrics stored with the model. These metrics are surfaced in reports and registry entries.

For practical usage, see the CLI guide and GUI cockpit; for theory, see `07-theory-and-background/chemometrics_and_ml_basics.md`.
