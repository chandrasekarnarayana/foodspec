# User Guide – Automated Analysis

FoodSpec supports highly automated “one-click” analysis in two paths:

<!-- GUI path removed -->

## Group B: CLI Auto-Analysis + Publish
- **Why:** Scriptable, reproducible runs for batch use.
- **How:**
  ```bash
  # Run protocol end-to-end
  foodspec-run-protocol \
    --input examples/data/oils.csv \
    --protocol examples/protocols/oil_basic.yaml \
    --output-dir runs/auto_oil_basic \
    --auto --report-level standard

  # Auto-generate narrative and figure panels
  foodspec-publish runs/auto_oil_basic/<timestamp> --fig-limit 6
  ```
  - For multi-input/harmonized runs: add multiple `--input` flags (e.g., oils + chips).
  - For HSI: use `hsi_segment_roi` protocol and the HSI example data.
- **Outputs:** run bundle with `report.txt/html`, `figures/`, `tables/`, `metadata.json`, `index.json`, `run.log`, and optionally `models/` (frozen pipelines).
- **Dry-run:** use `--dry-run` to validate/estimate without executing (helpful before large HSI/multi-input runs).

## Tips for best automation
- Always run validation via the protocol’s validation block; fix blocking errors before proceeding.
- Let protocols choose validation strategy by default; they auto-reduce CV folds when classes are tiny.
- Use HDF5 for multi-instrument/HSI to retain harmonization metadata.
- For repeated runs, organize CLI outputs under a project folder.

Cross-links: [cli_guide.md](cli.md), [cookbook_rq_questions.md](../theory/rq_engine_detailed.md), [validation_strategies.md](../05-advanced-topics/validation_strategies.md).
