# End-to-End Notebooks

- `examples/notebooks/01_oil_discrimination_basic.ipynb`
- `examples/notebooks/02_oil_vs_chips_matrix_effects.ipynb`
- `examples/notebooks/03_hsi_surface_mapping.ipynb`

Run all cells to see FoodSpec workflows end-to-end on synthetic data.
# End-to-end notebooks

FoodSpec ships runnable notebooks under `examples/notebooks/` to demonstrate full workflows with minimal setup.

## Notebooks
- **01_oil_discrimination_basic.ipynb**  
  Problem: oil authentication. Runs the oil discrimination protocol via Python/CLI, inspects figures/tables/narrative. Expected: confusion matrix, discriminative ratios, minimal panel.  

- **02_oil_vs_chips_matrix_effects.ipynb**  
  Problem: matrix effects. Uses oil + chips datasets to run the oil-vs-chips protocol; reviews divergence tables, effect sizes, and plots. Expected: matrix divergence tables/plots.  

- **03_hsi_surface_mapping.ipynb**  
  Problem: HSI segmentation/ROI analysis. Loads a synthetic HSI cube, runs segmentation → ROI → RQ, visualizes label maps and ROI spectra. Expected: label map, ROI spectra comparisons.  

## How to run
1. Install (core is sufficient; HSI uses matplotlib/seaborn included):
   ```bash
   pip install foodspec
   ```
2. Launch Jupyter or VS Code:
   ```bash
   python -m jupyter notebook
   ```
3. Open the notebook and run all cells. Run folders are typically created under `runs/`; the notebook points to outputs.

## What you should observe
- Data loading (CSV/HDF5/HSI) and protocol selection via Python/CLI calls.
- Bundle inspection (figures, tables, narrative).
- Interpretation of key outputs: discriminative markers, matrix effects, heating trends, HSI segmentation.

## Cross-links
- Getting started: [first-steps_cli.md](../01-getting-started/first-steps_cli.md)  
- Tutorials: [oil_discrimination_basic.md](oil_discrimination_basic.md)  
- Cookbook: [cookbook_rq_questions.md](../03-cookbook/cookbook_rq_questions.md)
