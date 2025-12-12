# Modeling-GUI: FoodSpec / RQ Workflow

This guide shows how to run FoodSpec’s Ratio-Quality analysis from the PyQt GUI.

## Steps (protocol-like)
1) **Load data**
   - Click “Load CSV” and pick your Raman/FTIR table (wide-format peaks or ratios).
   - The FoodSpec tab will display the file name and try to auto-map common columns.
2) **Map columns**
   - In the FoodSpec / RQ tab sidebar, select the columns for oil type, matrix (oil/chips), heating stage, replicate, sample ID.
3) **Choose preset / config**
   - Use the default Raman preset or load a JSON/YAML preset of peaks/ratios.
   - Toggle which analyses to run (stability, discrimination, heating trends, oil vs chips).
4) **Run automatic analysis**
   - Click “Run FoodSpec Auto-Analysis”.
   - A progress indicator and log will show the status.
5) **Review results**
   - **Overview**: executive summary.
   - **Stability**: CV table and top stable features plot.
   - **Discriminative Power**: p-values or feature importance plot/table.
   - **Heating Trends**: slopes/monotonicity table and bar plot.
   - **Oil vs Chips**: matrix comparison table and grouped bars.
   - **Report & Export**: full text report; save as .txt/.html or copy to clipboard.
6) **Export**
   - Save the report, and right-click/save plots from the tabs (matplotlib toolbar for zoom/pan).

## Automatic Analysis Protocol (what happens under the hood)
- Validates mappings and optionally auto-normalizes to a reference peak (e.g., 2720 cm⁻¹).
- Computes ratios, stability (CV/MAD), discriminative tests (ANOVA/Kruskal), heating trends (slopes/p), and oil-vs-chips divergence.
- Generates an executive summary + full text report (RQ1–RQ6 style) and key plots.
- Updates all GUI tabs from a single run object (no manual stitching needed).

## What to look for (UI guide in lieu of screenshots)
- **Overview tab**: executive summary, QC/validation notes; confirms protocol and validation strategy.
- **Stability tab**: table + bar plot of top stable ratios (CV/MAD).
- **Discriminative Power tab**: importance bars, sortable p-value/effect-size table, minimal marker panel summary.
- **Heating Trends tab**: slopes, Spearman ρ, monotonic flags; trend plot per ratio.
- **Oil vs Chips tab**: divergence tables/plots if matrix data present.
- **HSI tab**: mean/label maps and ROI summaries when using HSI protocols.
- **Report & Export tab**: full protocol text answer; save/copy buttons for .txt/.html.
