# FoodSpec Raman GUI quickstart

Use the GUI when you want a turnkey, publication-grade Raman analysis without touching code. It wraps `scripts/raman_workflow_foodspec.py` and adds automation, model artifacts, and download bundles while leaving the core package untouched.

## Install lightweight UI deps
```bash
pip install streamlit joblib
```
`joblib` ships with scikit-learn but is listed for clarity.

## Launch
```bash
streamlit run scripts/raman_workflow_gui.py --server.headless true
```

## What the GUI does
- Loads a Raman CSV (upload or path), runs the full RQ1–RQ14 workflow with safe defaults, and writes outputs to `results/<run_name>`.
- Creates tables, figures, summary report, and trained RF models (oil ID classifier + heating-stage regressor).
- Generates a zipped bundle plus `gui_run_metadata.json` describing the run.
- Auto-suggests research prompts (normalization robustness, top ratios, thermal markers, peak shifts) and shows key figures/tables inline.
- Exposes a feature inventory that mirrors README/docs capabilities so users can see what is covered.

## Best practices for users
- Leave defaults for a first pass; only tweak baseline/SG parameters if the spectra are very noisy.
- Use short, unique `run_name` values to keep outputs organized.
- After the run, download the report and bundle for archival; the models live in `results/<run_name>/models`.
