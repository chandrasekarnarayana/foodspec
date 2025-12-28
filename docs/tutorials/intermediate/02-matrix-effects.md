# Tutorial: Oil vs Chips Matrix Effects

- Data: `examples/data/oil_synthetic.csv`, `examples/data/chips_synthetic.csv`
- Notebook: `examples/notebooks/02_oil_vs_chips_matrix_effects.ipynb`
- Protocol: `examples/protocols/Chips_vs_Oil_MatrixEffects_v1.yml`
- Focus: Divergence markers, effect sizes, interpretation of ratios.
# Tutorial – Oil vs chips matrix effects

### What this tutorial covers
- **Problem:** Matrix effects—markers may behave differently in pure oils vs chips.  
- **Datasets:** `examples/data/oils.csv` and `examples/data/chips.csv` (matching peak/ratio columns, metadata).  
- **Protocol:** `examples/protocols/oil_vs_chips.yaml` (divergence analysis).

## Why it matters (theory)
Matrix components (starch/protein) can alter mean ratios, CV, and heating trends. Identifying matrix-robust vs matrix-sensitive markers is critical for QA. See [rq_engine_theory.md](../../theory/rq_engine_detailed.md).

<!-- GUI workflow removed; use CLI workflow below -->

## CLI workflow
```bash
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --input examples/data/chips.csv \
  --protocol examples/protocols/oil_vs_chips.yaml \
  --output-dir runs/oil_vs_chips_demo

foodspec-publish runs/oil_vs_chips_demo/<timestamp> --fig-limit 6
```
Open `figures/` for divergence plots and trend charts; `tables/` for oil_vs_chips summaries.

## Example figure (from run bundle)
![Matrix divergence](../../assets/figures/oil_vs_chips_divergence.png)

## How to read the results
- **Divergence tables**: look for significant differences (post-FDR) in mean, CV, or trends.  
- **Effect sizes** (e.g., Cohen’s d, slope deltas) quantify practical impact.  
- **Interpretation tags** (e.g., “stable in oil, unstable in chips”) highlight matrix sensitivity.  
- Prefer matrix-robust markers for cross-matrix QA; avoid markers that flip behavior between oil and chips.

## Cross-links
- Cookbook: [cookbook_rq_questions.md](../../theory/rq_engine_detailed.md) and [cookbook_preprocessing.md](../../methods/preprocessing/normalization_smoothing.md)  
- Theory: [rq_engine_theory.md](../../theory/rq_engine_detailed.md)
