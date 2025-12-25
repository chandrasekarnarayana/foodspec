# FoodSpec Integration Stock-Take

This document tracks end-to-end integration coverage across FoodSpec modules, CLIs, and experiment/YAML flows.

| Capability Area | Feature | Module location (path) | Python API entrypoint | CLI entrypoint (command + options) | YAML/Protocol access | Algorithms implemented | Metrics produced | Outputs | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| IO | Folder ingest → HDF5 | src/foodspec/cli.py#L376-L420 | `load_folder()` → `_default_preprocess_pipeline()` → `to_hdf5()` | `foodspec preprocess --input-folder --metadata-csv --output-hdf5 --modality --min-wn --max-wn` | N/A | ALS baseline, Savitzky–Golay, L2 vector norm, crop | N/A | HDF5 library file | Implemented | Uses default pipeline; advanced options via protocol/YAML, not CLI here |
| IO | CSV → HDF5 library | src/foodspec/cli.py#L860-L1011 | `load_csv_spectra()` → `create_library()` | `foodspec csv-to-library --format wide|long --wavenumber-column --sample-id-column --intensity-column --label-column --modality` | N/A | Layout conversion only | N/A | HDF5 library file | Implemented | Validates columns and modality |
| Preprocess | Protocol runner preprocessing | src/foodspec/cli_protocol.py | `ProtocolRunner.run()` → `PreprocessStep` | `foodspec-run-protocol --baseline-method --normalization-mode --spike-removal|--no-spike-removal` | `preprocess.params.*` (e.g., `baseline_method`, `normalization`, `spike_removal`) | Baseline (ALS, rubberband, polynomial), Savitzky–Golay smoothing, normalization (reference, vector, area, max), spike removal | `spikes_removed` column; preprocessing config in metadata | Run folder with tables/figures/metadata | Implemented | Default spike removal enabled if protocol does not specify; CLI override documented |
| Preprocess | Pipeline for raw spectra → peaks | src/foodspec/preprocessing_pipeline.py | `run_full_preprocessing(config)` | Via protocol runner and YAML | `preprocess.*` | Baseline (ALS/rubberband/polynomial), Savitzky–Golay, normalization, spike/cosmic-ray correction | `spikes_removed` per-spectrum counts; processed spectra columns | Processed DataFrame; optional peak extraction | Implemented | Smoothing window/polyorder from config; alignment optional |
| QC | Batch QC/novelty | src/foodspec/cli.py#L480-L509 | `generate_qc_report()` via `FoodSpec.qc()` | `foodspec qc --model-type oneclass_svm|isolation_forest --label-column --output-dir` | `qc.*` in exp.yml | OneClassSVM, IsolationForest (as per CLI options) | Outlier rate (`qc_outliers.outlier_rate`), drift score/trend slope; health aggregates | Report folder with `scores.csv`, figures | Implemented | CLI trains QC model and applies; confirms threshold in result |
| Features | Feature engine (peaks/bands) | src/foodspec/core/api.py#L293-L351 | `FoodSpec.features(preset|specs)` | Via exp.yml → features preset/specs | `features.*` in exp.yml | Bands and peak extraction via `FeatureEngine` | `n_features`; features table diagnostic | Features table in diagnostics | Implemented | Presets quick/standard provide minimal specs |
| RQ | RatioQuality engine | src/foodspec/protocol_engine.py#L153-L222 | `RatioQualityEngine(peaks, ratios)` | Via protocol runner (RQ step) | `rq_analysis.*` | MANOVA/Tukey/Games–Howell (as part of RQ engine) | Balanced accuracy, per-class recall, ROC-AUC (when available) | Metrics JSON/CSV, confusion matrix | Implemented | Validation metrics computed when class sizes allow |
| Chemometrics | Classifier factory | src/foodspec/chemometrics/models.py | `make_classifier(model_name)` | Via `foodspec oil-auth` / domains / run-exp | `modeling.*` in exp.yml | RF, SVM (linear/rbf), LogisticRegression, KNN, GradientBoosting, MLP (plus optional XGB/LGBM) | `accuracy`, `precision`, `recall`, `f1`; optional `roc_auc`, `average_precision` | HTML/markdown report, metrics CSV, confusion_matrix.png | Implemented | Metrics via `chemometrics.validation.compute_classification_metrics` |
| Heating | Oxidation trajectory | src/foodspec/cli.py#L465-L478; src/foodspec/core/api.py#L560-L680 | `FoodSpec.analyze_heating_trajectory(...)` | `foodspec heating --time-column --output-dir` | `moats.heating_trajectory.*` | Linear trend fit; ANOVA; stage classification (optional) | Trend slopes/intercepts, ANOVA table presence; stage classification metrics | Report folder with ratios CSV, trend_models CSV, ratio_vs_time plot | Implemented | Stage classification and shelf-life optional |
| Stats | Spectral similarity metrics | src/foodspec/features/library.py; src/foodspec/library_search.py | `LibraryIndex.search(metric=...)` | `foodspec library-auth --metric --top-k`; `foodspec-library-search --metric --k` | N/A | Euclidean, Cosine, Pearson, SID, SAM | Similarity table: `query_index`, `query_id`, `library_index`, `distance`, `rank`, `lib_*` metadata columns | Similarity table CSV; overlay figure | Implemented | Core API also records overlay figure in bundle |
| Workflows | Oil authentication | src/foodspec/cli.py#L420-L465 | `FoodSpec(...).features().train()` via workflow app | `foodspec oil-auth --label-column --cv-splits --output-report --save-model` | `modeling.*` | Classifier training with CV | CV metrics mean; confusion matrix | Report HTML/MD, metrics CSV, confusion_matrix.png | Implemented | Saves model via registry when requested |
| Workflows | Domains (dairy/meat/microbial) | src/foodspec/cli.py#L509-L662 | Domain apps | `foodspec domains --type --label-column --classifier-name --cv-splits --output-dir --save-model` | `modeling.*` | Same suite as oil-auth | CV metrics; feature importances (if available) | Report folder | Implemented | Wrapper over core workflows |
| Repro | Experiment YAML engine | src/foodspec/repro/experiment.py; src/foodspec/cli.py#L736-L820 | `ExperimentEngine.from_yaml()`; `FoodSpec` chain | `foodspec run-exp --dry-run --output-dir --artifact-path` | `dataset`, `preprocessing`, `qc`, `features`, `modeling`, `outputs`, `moats.*` | Declarative wiring across modules | `config_hash`, per-step hashes, dataset hash; various module metrics | Exported run folder; single-file `.foodspec` artifact | Implemented | Dry-run prints summary and hashes |
| Report | Publish bundle | src/foodspec/cli_publish.py | `save_markdown_bundle(run_dir, out, ...)` | `foodspec-publish --run-dir --out --fig-limit --profile` | N/A | Figure selection profiles; narrative assembly | Markdown panel; summary.json | Methods/figures bundle directory | Implemented | Profiles: quicklook/qa/standard/publication |
| Deploy | Single-file artifact predict | src/foodspec/cli_predict.py; src/foodspec/artifact.py | `load_artifact()` → `Predictor.predict()` | `foodspec-predict --model --input/--input-dir --output/--output-dir` | N/A | Model prediction (via embedded artifact pipeline) | Adds `qc_do_not_trust`, `qc_notes` when probabilities present | predictions.csv per input | Implemented | QC guard applied when proba columns exist |
| CLI | Protocol benchmarks | src/foodspec/cli.py#L980-L1011 | `run_protocol_benchmarks()` | `foodspec protocol-benchmarks --output-dir --random_state --config` | N/A | Benchmarks over public datasets | Summary JSON (aggregate metrics) | Benchmark folder | Implemented | Writes run metadata |

## Smoke Test (5 commands)

```bash
# 1) Protocol run (spike removal ON)
foodspec-run-protocol --input examples/data/oil_synthetic.csv --protocol examples/protocols/EdibleOil_Classification_v1.yaml --output-dir protocol_runs --spike-removal

# 2) Protocol run (spike removal OFF)
foodspec-run-protocol --input examples/data/oil_synthetic.csv --protocol examples/protocols/EdibleOil_Classification_v1.yaml --output-dir protocol_runs --no-spike-removal

# 3) QC / novelty detection (create demo HDF5 then run CLI)
python -c "from foodspec.data.loader import load_example_oils; from foodspec.io.exporters import to_hdf5; ds=load_example_oils(); to_hdf5(ds, 'examples/data/oils_demo.h5')" && \
foodspec qc examples/data/oils_demo.h5 --model-type isolation_forest --output-dir protocol_runs/qc

# 4) Library search (one-row query vs library from examples)
head -n 1 examples/data/oil_synthetic.csv > /tmp/header.csv && head -n 2 examples/data/oil_synthetic.csv | tail -n 1 >> /tmp/query_one_row.csv && \
foodspec-library-search --query /tmp/query_one_row.csv --library examples/data/oil_synthetic.csv --label-col oil_type --k 3 --metric cosine --overlay-out protocol_runs/library_overlay.png

# 5) YAML experiment (dry run)
foodspec run-exp examples/configs/oil_auth_quickstart.yml --dry-run
```

## Definition of Done (Integration)

- Metrics present: core steps emit concrete fields (e.g., `accuracy`, `precision`, `recall`, `f1`, `roc_auc` where available; QC outlier/drift; similarity table columns; `spikes_removed`).
- Provenance recorded: `RunRecord` contains per-step hashes, dataset hash, environment capture, and output paths.
- CLI parity: user can run QC, protocol execution, domain/oil workflows, publish reports, predict from artifacts, and perform library search.
- Docs updated: CLI help for protocol runner spike toggles and overrides, quickstart for protocol execution, and this integration stock-take.
