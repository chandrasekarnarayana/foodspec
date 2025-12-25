# Protocol benchmarks

## Purpose
Protocol benchmarks validate the robustness and generality of the FoodSpec protocol across public datasets. They are a reference point to judge whether new preprocessing or models are “good enough” to be called a protocol.

## Command
```bash
foodspec protocol-benchmarks --output-dir runs/protocol_benchmarks
```

Artifacts (timestamped run folder):
- `classification_metrics.json` + `classification_confusion_matrix.csv`
- `mixture_metrics.json` (regression on mixtures)
- `report.md` summary and `run_metadata.json` (environment/version info)

## Interpretation
- Classification: look at accuracy/F1 and confusion matrix; high scores suggest strong discriminative power on the benchmark oil dataset.
- Mixture: R² and RMSE indicate fraction-estimation quality; use as a baseline before deploying new pipelines.
- If errors are present (e.g., missing datasets), the summary will include clear messages—benchmarks should only be compared when datasets are available.

## When to use
- Before publishing or claiming protocol stability.
- After major pipeline changes to ensure regressions are not introduced.
- To compare alternate preprocessing/model choices against a fixed reference.
