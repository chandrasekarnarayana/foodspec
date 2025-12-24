# Tutorial: Oil Discrimination (Basic)

- Data: `examples/data/oil_synthetic.csv`
- Notebook: `examples/notebooks/01_oil_discrimination_basic.ipynb`
- Protocol: `examples/protocols/EdibleOil_Classification_v1.yml`
- Steps: Load → Run protocol → View report/tables → Publish with `foodspec-publish`.
# Tutorial – Oil discrimination (basic)

### What this tutorial covers
- **Problem:** Authenticate edible oils (e.g., VO/PO/OO/CO) using Raman ratios; see why discrimination matters in QA.  
- **Dataset:** `examples/data/oils.csv` (or HDF5), with oil_type, heating_stage (optional), replicate/batch.  
- **Protocol:** `examples/protocols/oil_basic.yaml` (ratiometric features, validation, minimal panel).

## Why it matters (theory)
Authenticating oils protects against adulteration and ensures quality. Discriminative ratios (e.g., carbonyl/unsaturation bands) separate oil types. Validation (batch-aware CV) checks generalization; see [rq_engine_theory.md](../07-theory-and-background/rq_engine_theory.md) and [chemometrics_and_ml_basics.md](../07-theory-and-background/chemometrics_and_ml_basics.md).

<!-- GUI workflow removed; use CLI workflow below -->

## CLI workflow
```bash
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/oil_basic.yaml \
  --output-dir runs/oil_basic_tutorial

foodspec-publish runs/oil_basic_tutorial/<timestamp> --fig-limit 6
```
Check `figures/` for confusion matrix and discriminative barplots; `tables/` for discriminative and minimal panel summaries.

## Example figures (from run bundle)
![Confusion matrix](../assets/figures/oil_confusion.png)
![Top discriminative ratios](../assets/figures/oil_discriminative.png)
![Minimal panel accuracy](../assets/figures/oil_minimal_panel.png)
![Stability](../assets/figures/oil_stability.png)

## How to read the results
- **Confusion matrix/balanced accuracy:** off-diagonals show misclassifications; balanced accuracy corrects imbalance.  
- **Discriminative ratios/effect sizes:** look for high importance or low FDR-corrected p-values.  
- **Minimal marker panel:** smallest feature set meeting target accuracy—useful for QA panels.  
- **Stability (CV/MAD):** lower CV/MAD indicates more reproducible markers.

## Cross-links
- Cookbook recipes: [cookbook_rq_questions.md](../03-cookbook/cookbook_rq_questions.md)  
- Theory: [rq_engine_theory.md](../07-theory-and-background/rq_engine_theory.md)
