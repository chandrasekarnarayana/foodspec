## Protocol Cookbook

### EdibleOil_Classification_v1
- Purpose: RQ1–RQ3 discrimination of oils.
- Steps: preprocess (reference normalization), rq_analysis with key ratios (1742/2720, 1652/2720, etc.), output bundle.

### EdibleOil_Heating_Stability_v1
- Purpose: RQ2–RQ4 thermal degradation and stability markers.
- Steps: preprocess, rq_analysis (heating trends enabled), output.

### Chips_vs_Oil_MatrixEffects_v1
- Purpose: matrix divergence between pure oil and chips.
- Steps: preprocess, rq_analysis with oil_vs_chips summary, output.

### MinimalMarkerPanel_Screening_v1
- Purpose: RQ5 minimal feature panel for QA.
- Steps: preprocess, rq_analysis with minimal panel search, output.

### HSI ROI to RQ (example)
- Purpose: segment hyperspectral cube, extract ROI spectra, feed into RQ.
- Steps: hsi_segment -> hsi_roi_to_1d (peaks/ratios) -> rq_analysis -> output.

### Extending via plugins
- Protocols/vendor loaders can be added via plugins. See `examples/plugins/`.
- Install a plugin (development mode): `pip install -e examples/plugins/plugin_example_protocol`
- List discovered plugins: `foodspec-plugin list`

### Notebook references (try these first)
- `examples/notebooks/01_oil_discrimination_basic.ipynb`: basic oil discrimination on synthetic data.
- `examples/notebooks/02_oil_vs_chips_matrix_effects.ipynb`: oil vs chips divergence demo.
- `examples/notebooks/03_hsi_surface_mapping.ipynb`: HSI segmentation → ROI → RQ.

- ### Example SOP – Oil Authentication (FoodSpec)
- Protocol: `examples/protocols/EdibleOil_Classification_v1.yml`.
- Replicates: aim for ≥3 spectra per oil type (per batch) for defensible statistics; more for large studies.
- Run via CLI with defaults (5-fold CV, reference normalization).
- Include in reports: stability table/plot, discriminative ratios (top ANOVA/importance), minimal panel summary, validation metrics block, and figure panel from `foodspec-publish`.
