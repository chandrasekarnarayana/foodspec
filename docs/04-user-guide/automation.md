# User Guide – Automated Analysis

FoodSpec supports highly automated “one-click” analysis in two paths:

## Group A: GUI Auto-Analysis (one-click)
- **Why:** Fastest path for non-developers; visual validation and progress.
- **How:**
  1. Launch `foodspec-gui` (wizard or cockpit).
  2. Load data (CSV/HDF5); confirm column mapping.
  3. Select a protocol preset (oil discrimination, thermal stability, oil vs chips, HSI).
  4. Click **Validate**, then **Run**. All steps (preprocess → harmonize → QC → RQ → bundle) execute automatically.
  5. When finished, the GUI shows the run folder link with report/figures/tables/models.
- **Where to review:** Overview, Stability, Discriminative, Trends, Oil vs Chips, HSI tabs; Report & Export for text/HTML.

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
- Always run validation (GUI button or CLI validation block); fix blocking errors before proceeding.
- Let protocols choose validation strategy by default; they auto-reduce CV folds when classes are tiny.
- Use HDF5 for multi-instrument/HSI to retain harmonization metadata.
- For repeated runs, keep a run history (GUI) or organize CLI outputs under a project folder.

Cross-links: [gui_cockpit_guide.md](gui_cockpit_guide.md), [cli_guide.md](cli_guide.md), [cookbook_rq_questions.md](../03-cookbook/cookbook_rq_questions.md), [validation_strategies.md](../05-advanced-topics/validation_strategies.md).
