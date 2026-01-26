# Mindmap Compliance Report

## Summary
Coverage: high across core spectroscopy workflows; structural compliance: high (mindmap namespaces own primary logic, legacy adapters remain); docs compliance: medium-high (link fixes in place; JOSS scaffold restored).

Key improvements in this refactor:
- Enforced a shared CLI artifact contract (`manifest.json`, `run_summary.json`, `logs/run.log`) with standardized exit codes.
- Tightened IO validators for axis/intensity/metadata schema checks and enforced validation on ingestion.
- Centralized QC policy enforcement and standardized QC reporting outputs.
- Expanded model and dataset cards into real run artifacts with reproducibility metadata.
- Restored a minimal JOSS scaffold under `docs/joss/`.

## Artifact Contract (CLI Output Standards)
All CLI runs emit:
- `manifest.json`
- `run_summary.json`
- `logs/run.log`

Exit codes:
- `0` success
- `2` validation/schema failure
- `3` QC failure when required by policy/protocol
- `4` runtime error

## Compliance Table

| **Mindmap Node** | **Code Modules** | **Public API** | **CLI** | **Docs** | **Examples** | **Status** | **Notes / Next actions** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core Design Philosophy | `foodspec.core.philosophy` | `GOALS`, `NON_GOALS`, `feature_map()` | `foodspec about` | `docs/concepts/design_philosophy.md` | N/A | ✅ | Mapped goals/non-goals and mindmap links are explicit. |
| Data Objects | `foodspec.data_objects.*` | `Spectrum`, `FoodSpectrumSet`, `SpectralDataset` | `foodspec io validate` | `docs/concepts/data_objects.md` | `examples/advanced/spectral_dataset_demo.py` | ✅ | Primary home established; core imports are shims. |
| Data Extraction | `foodspec.io`, `foodspec.io.loaders` | `read_spectra`, `detect_format`, `load_csv_spectra` | `foodspec io validate` | `docs/user-guide/vendor_io.md` | `examples/quickstarts/phase1_quickstart.py` | ✅ | Axis/intensity/metadata schema checks enforced on ingestion. |
| Programming Engine | `foodspec.engine.preprocessing`, `foodspec.engine.pipeline` | `PreprocessPipeline`, `Step` | `foodspec preprocess run` | `docs/methods/preprocessing/baseline_correction.md` | `examples/validation/validation_preprocessing_baseline.py` | ⚠ | Legacy `preprocess/` remains; migrate remaining logic. |
| Quality Control System | `foodspec.qc.*`, `foodspec.qc.policy` | `compute_health_scores`, `QCPolicy` | `foodspec qc spectral|dataset` | `docs/concepts/qc_system.md` | `examples/quickstarts/qc_quickstart.py` | ✅ | Policy enforced in CLI and QC reports saved for each run. |
| Feature Engineering | `foodspec.features.*`, `foodspec.features.selection` | `PeakFeatureExtractor`, `compute_ratios`, `compute_minimal_panel` | `foodspec features extract` | `docs/methods/preprocessing/feature_extraction.md` | `examples/validation/validation_peak_ratios.py` | ✅ | Minimal panel exposed from selection helpers. |
| Modeling & Validation | `foodspec.modeling.*`, `foodspec.modeling.validation`, `foodspec.modeling.evaluation` | `fit_predict`, `make_classifier`, `compute_classification_metrics` | `foodspec train`, `foodspec evaluate`, `foodspec model predict` | `docs/methods/chemometrics/model_evaluation_and_validation.md`, `docs/user-guide/validation.md` | `examples/validation/validation_chemometrics_oils.py` | ⚠ | Nested CV + LOBO/LOSO supported; continue migration from legacy `ml/`. |
| Trust & Uncertainty | `foodspec.trust.*` | `PlattCalibrator`, `IsotonicCalibrator`, `MondrianConformalClassifier`, `apply_abstention_rules`, `RegulatoryReadiness` | `foodspec trust fit|conformal|abstain` | `docs/concepts/trust_uncertainty.md` | `examples/new-features/uncertainty_demo.py` | ⚠ | Trust artifacts emitted per-run (calibration, conformal, abstention, readiness); consider wiring into training workflows. |
| Visualization & Reporting | `foodspec.viz`, `foodspec.reporting` | `ReportBuilder`, `PDFExporter` | `foodspec report` | `docs/help/reporting_infrastructure.md` | `examples/new-features/pdf_export_demo.py` | ✅ | Report paths and artifacts are stable. |
| API Accessibility | `foodspec.cli`, `foodspec.__init__` | `foodspec` namespace, CLI groups | `foodspec --help` | `docs/getting-started/quickstart_cli.md` | `examples/quickstarts/phase1_quickstart.py` | ✅ | Mindmap-aligned CLI groups added with manifests. |
| Documentation & JOSS Readiness | `docs/`, `CITATION.cff` | N/A | N/A | `docs/index.md`, `docs/reproducibility.md`, `docs/joss/paper.md` | `docs/examples/index.md` | ⚠ | Minimal JOSS scaffold restored; expand paper before release. |

## Deprecations
- `foodspec.core.dataset` → `foodspec.data_objects.spectra_set` (through v2.0.0)
- `foodspec.core.spectral_dataset` → `foodspec.data_objects.spectral_dataset` (through v2.0.0)
- `foodspec.data_objects.spectraset` → `foodspec.data_objects.spectra_set` (through v2.0.0)
- `foodspec.preprocess.engine` → `foodspec.engine.preprocessing.engine` (through v2.0.0)
- `foodspec.ml.nested_cv` → `foodspec.modeling.validation.splits` (through v2.0.0)
- `foodspec.chemometrics.validation` → `foodspec.modeling.validation.metrics` (through v2.0.0)
- `foodspec.validation` → `foodspec.modeling.validation` (through v2.0.0)

## Doc link fixes
- `docs/developer-guide/BACKWARD_COMPAT_EXAMPLES.md` → `docs/developer-guide/COMPATIBILITY_PLAN.md`
- `docs/developer-guide/PHASE_0_CHECKLIST.md` → `docs/developer-guide/ENGINEERING_RULES.md`, `docs/developer-guide/GIT_WORKFLOW.md`
- `docs/reference/method_comparison.md` → inline definitions (removed broken internal link)
- `docs/user-guide/export_quickstart.md` → `examples/new-features/export_demo.py`
- `docs/user-guide/pdf_export_quickstart.md` → `examples/new-features/pdf_export_demo.py`
