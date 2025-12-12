## FoodSpec Architecture

- **IO & Harmonization**: `spectral_io.py`, vendor stubs, CSV/HDF5 loaders; `spectral_dataset.py` for 1D spectra and `HyperspectralDataset` for cubes; simple alignment utilities.
- **Preprocessing**: `PreprocessingConfig` + `preprocessing_pipeline.py` (baseline, smoothing, normalization, peak extraction) feeding peak tables for analysis.
- **RQ Engine**: `rq.py` (stability, discrimination, trends, oil-vs-chips divergence, minimal panel, clustering, report generator).
- **Protocol Engine**: `protocol_engine.py` with step registry (preprocess, harmonize, hsi_segment, hsi_roi_to_1d, rq_analysis, qc_checks, output) executing YAML/JSON protocols and bundling outputs via `output_bundle.py`.
- **CLI**: `foodspec-run-protocol`, `foodspec-predict`, `foodspec-registry`, `foodspec-publish`, `foodspec-plugin`.
- **GUI**: PyQt cockpit (`scripts/foodspec_protocol_cockpit.py`) and wizard; wraps protocol runner.
- **Docs/Examples**: `examples/protocols/` for domain presets; `examples/plugins/` for community extensions.

## Registry and Provenance
- `FeatureModelRegistry` (`registry.py`) stores runs/models: dataset hash/file list, protocol name/version, preprocessing config, validation strategy, features, model path/type, metrics, provenance (timestamp/user/tool version).
- CLIs: `foodspec-registry` (list/query), `foodspec-plugin` (list/install).
- Models can auto-register on save if `FOODSPEC_REGISTRY` is set.
