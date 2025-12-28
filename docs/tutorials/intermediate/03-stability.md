# Tutorial: Thermal Stability Tracking

- Data: oil/chips synthetic (or your own heating series)
- Protocol: `examples/protocols/EdibleOil_Heating_Stability_v1.yml`
- Outputs: Heating trend tables/plots, stability rankings, interpretation tables.
# Tutorial – Thermal stability tracking

### What this tutorial covers
- **Problem:** Thermal degradation across heating stages; find heating-sensitive markers.  
- **Dataset:** `examples/data/oils.csv` (or chips) with `heating_stage`.  
- **Protocol:** `examples/protocols/oil_heating.yaml` (trend analysis, monotonicity).

## Why it matters (theory)
Heating alters oxidation/unsaturation signatures. Trend analysis (linear slopes, Spearman ρ) shows whether ratios increase/decrease with heating. See [rq_engine_theory.md](../../theory/rq_engine_detailed.md).

<!-- GUI workflow removed; use CLI workflow below -->

## CLI workflow
```bash
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/oil_heating.yaml \
  --output-dir runs/oil_heating_demo

foodspec-publish runs/oil_heating_demo/<timestamp> --fig-limit 6
```
Open `figures/trend_*` for ratio vs heating stage; `tables/heating_trend_summary.csv` for slopes/ρ/p-values/monotonicity.

## Example figures (from run bundle)
![Heating trends](../../assets/figures/heating_trend.png)

## How to read the results
- **Slopes (p < 0.05 FDR):** heating-sensitive markers.  
- **Spearman ρ:** near ±1 indicates monotonic change.  
- **Stability ranking:** lower abs(slope) = more stable spectral fingerprint.  
- Choose markers with consistent trend direction across replicates/batches for QA.

## Cross-links
- Cookbook: [cookbook_rq_questions.md](../../theory/rq_engine_detailed.md)  
- Theory: [rq_engine_theory.md](../../theory/rq_engine_detailed.md)
