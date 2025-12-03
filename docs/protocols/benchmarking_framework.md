# Protocols: Benchmarking Framework

This framework standardizes how to compare preprocessing and modeling pipelines across spectral datasets. It leverages FoodSpec CLI/API to generate metrics, plots, and run metadata for fair comparisons.

## Goals
- Assess robustness and generality of pipelines (authentication, mixture regression, QC).
- Enable reproducible comparisons (fixed seeds, documented configs).
- Produce sharable reports (metrics.json, confusion matrices, plots, run_metadata.json).

```mermaid
flowchart TD
  A[Select datasets/tasks] --> B[Define pipelines (preproc+model variants)]
  B --> C[Run benchmarks (CLI/API)]
  C --> D[Collect artifacts (metrics, plots, metadata)]
  D --> E[Analyze & compare (tables/figures)]
```

## How to run (CLI)
Use `foodspec protocol-benchmarks --output-dir runs/benchmarks`.
- Internally runs classification and mixture benchmarks (using public/example loaders or synthetic fallbacks).
- Outputs: `classification_metrics.json`, `mixture_metrics.json`, plots, `run_metadata.json`.
- See `benchmarks/benchmark_oil_authentication.py` for a scriptable example.

## How to run (scripts)
- `benchmarks/benchmark_oil_authentication.py`: runs oil-auth pipeline on example/public data; saves metrics, confusion matrix, PCA plots, run metadata.
- (Add similar scripts for heating/QC as needed.)

## Statistical comparisons across pipelines
- Use ANOVA/t-tests on performance metrics (e.g., macro F1 across preprocessing variants).
- For multiple configurations, collect metrics per fold/config, then test differences:
```python
import pandas as pd
from foodspec.stats import run_anova

df = pd.DataFrame({"f1": [0.8,0.82,0.83,0.75,0.76,0.77],
                   "pipeline": ["A","A","A","B","B","B"]})
res = run_anova(df["f1"], df["pipeline"])
print(res.summary)
```
- Interpret whether differences are statistically meaningful; report effect sizes when possible.

## Designing a benchmark
- **Datasets:** Choose representative tasks (oil auth, mixtures, QC). Ensure wavenumbers align.
- **Pipelines:** Vary baseline methods (ALS/rubberband), normalization (L2/SNV/MSC), models (RF/SVM/PLS).
- **Validation:** Stratified CV for classification; train/test or CV for regression; fix random seeds.
- **Metrics:** Classification (accuracy, macro F1, confusion matrix), regression (RMSE, MAE, R², residuals).

## Artifacts and metadata
- Always store: metrics.json, run_metadata.json (Python version, foodspec version, model params, seeds), plots (confusion matrices, residuals/PCA).
- Prefer YAML configs for pipeline definitions; log them with runs.

## Reporting
- Summarize metrics in tables; show key plots.
- Discuss robustness: variance across folds/seeds; sensitivity to preprocessing variants.
- Provide configs and metadata for reproducibility (see [Reproducibility checklist](reproducibility_checklist.md)).

## Further reading
- [MethodsX protocol](methodsx_protocol.md)
- [Oil authentication workflow](../workflows/oil_authentication.md)
- [Model evaluation](../ml/model_evaluation_and_validation.md)
