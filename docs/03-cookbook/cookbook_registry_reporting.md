# Cookbook: Registry & Reporting

- Registry: auto-log runs/models by setting `FOODSPEC_REGISTRY`, query via `foodspec-registry`.
- Reporting: use `foodspec-publish` to generate methods text + figure panel.
- Include stability/discrimination/trend plots and validation metrics in reports.
# Cookbook – Registry & reporting

Recipes for querying runs/models and generating report-ready outputs.

## Query the registry for a model/protocol
- **CLI:**  
  ```bash
  foodspec-registry list
  foodspec-registry query --protocol oil_basic
  ```
- **Python:**  
  ```python
  from foodspec.registry import FeatureModelRegistry
  reg = FeatureModelRegistry("~/.foodspect_registry.db")
  models = reg.query_models(protocol_name="oil_basic")
  ```
- **What you get:** protocol name/version, dataset hash/file list, preprocessing summary, validation strategy, model path/type, metrics.

## Generate Methods text and figure panels
- **CLI:**  
  ```bash
  foodspec-publish run_folder --fig-limit 6
  ```
  Reads `metadata.json` + `index.json`, selects key figures (stability, discrimination, trend, confusion/ROC). Produces narrative text (Methods-style) and optional PDF/Markdown.  
  - `--fig-limit` caps panels; `--include-all-figures` exports everything as supplementary material.

## Inspect provenance for a specific model
- **Python:**  
  ```python
  entry = reg.query_models(model_path="path/to/frozen_model.pkl")
  print(entry.provenance)
  ```
- **Why:** Ensures reproducibility and auditability for QA or reports; check preprocessing, features, validation strategy, metrics.
