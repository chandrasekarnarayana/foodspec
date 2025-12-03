# Developer Notes / Coding Standards

<img src="../assets/foodspec_logo.png" alt="FoodSpec logo" width="200" />

## 1. Purpose of This Document
- Provide clear guidance for contributors and maintainers on how to write, document, and test foodspec code.
- Support long-term maintainability, clarity, and reproducibility across preprocessing, feature extraction, ML, stats, and workflows.
- Audience: contributors, maintainers, and students learning scientific Python and chemometrics in food spectroscopy.
- Documentation scope: domain workflow pages are canonical; tutorial stubs have been removed from navigation to avoid duplication. Keep new content consolidated in the workflow files.

## 2. Coding Principles and Standards
- **Clarity over cleverness**: Prefer readability, explicitness, and traceability over terse or obscure constructs.
- **Consistency**: Use consistent naming, folder structure, docstring style (Google/NumPy), and type hints. Keep function behavior predictable.
- **Structure & Modularity**: Keep modules focused (preprocess, features, ml, stats, workflows, io, utils). Each submodule has a clear responsibility.
- **Error Handling**: Raise explicit exceptions with actionable messages; avoid silent failures.
- **Type Hints & Data Contracts**: Use type hints everywhere; document expected shapes (e.g., `(n_samples, n_features)`, `(n_samples, n_wavenumbers)`).
- **Pure Functions When Possible**: Avoid hidden side-effects; make transformations explicit and testable.
- **Small Public API, Larger Private Internals**: Export a clean set of symbols in each module’s `__init__.py`; keep helpers private when not user-facing.

## 3. Documentation Standards
- Every user-facing function must include:
  - Clear docstring with summary, parameters, returns, assumptions/limitations.
  - A minimal code example using realistic shapes/inputs.
- Every concept should appear as: theory → code example → figure → workflow → API reference.
- Narrative docs must explain the “why,” not just the “how,” and connect to food spectroscopy use cases.
- **Docstring style:** Use NumPy or Google style consistently; include type hints. Add “See also” pointing to narrative chapters/workflows when useful.

## 4. Testing Standards
- Use pytest; keep tests small, isolated, and fast.
- Favor synthetic data for reliability and speed.
- Every new feature should ship with:
  - At least one unit test.
  - One end-to-end or integration test when relevant (e.g., CLI workflow or pipeline).
- Keep reproducibility in tests:
  - Fix random seeds where stochastic components are used.
  - Prefer mocks for optional vendor dependencies (SPC/OPUS) to avoid heavyweight installs.
  - Avoid real proprietary data; keep fixtures synthetic or clearly licensed.

## 5. File & Folder Conventions
- `src/foodspec/` : all library code (datasets, preprocessing, features, ml, stats, workflows, io, utils).
- `docs/` : user-facing documentation.
- `docs/examples/` : scripts/notebooks that generate figures or extended examples.
- `docs/assets/` : saved PNG/SVG images used in docs.
- `tests/` : pytest suite; mirrors src layout when possible.
- `docs/dev/` : internal notes (e.g., this page, design documents).

## 6. Scientific & Ethical Responsibility
- Transparency and reproducibility are core goals: users must trace every preprocessing, modeling, or statistical decision.
- Figures and results must be honest representations of data; no hidden preprocessing steps.
- Make all transformations explicit in code, configs, and reports.

## 7. Versioning & Change Philosophy
- Keep core APIs (preprocessing, stats, workflows) stable; document any breaking changes.
- Introduce new features with examples and tests.
- Deprecations must include warnings and migration notes.
- Optional dependencies (vendor readers, DL) must degrade gracefully with clear ImportErrors and doc notes.

## 8. How to Contribute Code
- Fork → branch → implement → document → test → pull request.
- Follow coding/documentation standards; include descriptive commit messages.
- Ensure lint/tests/docs pass before opening a PR.
- Run locally:
  - `pytest` (with coverage target ≥80%).
  - `mkdocs build` to catch doc/link issues.
  - (Optional) `ruff` or other linters if configured.

## 9. Architecture overview (high level)
- **Data layer:** `FoodSpectrumSet` / `HyperSpectralCube`, HDF5 libraries, I/O (CSV/JCAMP/vendor) normalize all inputs to a common representation.
- **Analysis layer:** preprocessing (baseline, smoothing, normalization, scatter/ATR/atmos), features (peaks/bands/ratios), stats (parametric/nonparametric/robustness), metrics, ML/chemometrics.
- **Workflow/CLI layer:** domain apps (oils, heating, mixtures, QC, hyperspectral), CLI commands, reporting/export helpers, reproducibility configs.

## 10. Developer Roadmap (High-Level)
- Vendor IO: continue to harden with mocks/fixtures and worked examples for SPC/OPUS/JCAMP; keep graceful fallbacks for missing optional deps.
- Coverage & CI: coverage is >90% (IO/text/vendor, stats/preprocess/viz branches now exercised); lightweight lint CI (ruff) added—keep both green on PRs (run `pytest --cov` + `ruff check` locally).
- Interpretability: RF importances, PLS loadings, and domain visuals are embedded in docs/workflows (see assets `rf_feature_importance.png`, `pls_loadings.png`); keep adding domain-specific figures as data permit.
- Workflows & calibration: workflows are canonical (tutorial stubs removed); keep expanding examples and calibration/regression utilities. DL remains optional—always compare against classical baselines and state cautions.
- Reproducibility/reporting: extend metadata/config patterns to new domains (dairy, spices, beverages) and add real-data tutorials when licensing permits.
- Diagrams: maintain consistent staged diagrams (data → preprocess → features → model/stats → report) across workflows; keep flowcharts readable and updated when workflow steps change.

## 11. Diagram / flowchart standards
- Use a consistent staged structure in workflow diagrams: **Data → Preprocess → Features → Model/Stats → Report**.
- Prefer left-to-right flowcharts (Mermaid `flowchart LR`) with grouped stages and concise labels; avoid cramped nodes.
- When workflow steps change, update the corresponding diagram and ensure text and links match.
- Keep colors/icons consistent with existing diagrams; avoid overly dense styling that hurts readability.
