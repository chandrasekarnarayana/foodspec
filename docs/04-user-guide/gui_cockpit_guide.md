# User Guide – GUI cockpit & wizard

Why it matters: the GUI provides guided, visual workflows for food scientists—minimal configuration, clear validation, and easy access to plots/tables.

## Cockpit layout (what you see)
- **Left sidebar**: Data & Mapping (file paths, column mapping), Analysis Config (protocol selector, checkboxes for stability/discrimination/trends/oil-vs-chips, validation strategy), Run buttons (Validate, Run).
- **Main tabs**:
  - Overview (executive summary, QC warnings, validation strategy).
  - Stability, Discriminative Power, Heating Trends, Oil vs Chips, Minimal Panel, HSI (if applicable).
  - Report & Export (text report viewer, save buttons).
- **Bottom/status**: Log/progress area with cancel button.
- **Run history**: panel/list of previous runs with links to run folders.

## Mini-workflow (cockpit)
1) **Load data** in Data & Mapping (e.g., `examples/data/oils.csv`); confirm column mapping.  
2) **Select protocol** (e.g., oil discrimination basic).  
3) **Validate**: click Validate; fix blocking errors from the dialog.  
4) **Run**: click Run; monitor progress; cancel if needed.  
5) **Review tabs** (Discriminative → confusion matrix/ratios; Stability → CV/MAD; Trends → slopes; HSI → label maps).  
6) **Open run folder** via run history link for `report.txt/html`, `figures/`, `tables/`, `metadata.json`, `index.json`.

## Wizard vs cockpit
- **Wizard**: guided, step-by-step with breadcrumbs (Load → Validate → Configure → Estimate runtime → Run → Review). Best for new users and SOP-like runs.
- **Cockpit**: direct access to all controls and tabs; best for power users needing quick iterations or model application.

## Project concept (multi-dataset)
- You can load multiple files (oil + chips, multi-instrument) into a project, assign roles (matrix, instrument, batch), and run harmonized protocols. Run history groups runs per project.

## Run history
- Accessible in the cockpit sidebar; shows timestamp, protocol, status, and run folder links.
- You can re-open a run folder to view reports/figures or apply a frozen model.

## Controls and options (checkboxes/dropdowns)
- **Protocol selector**: built-in and plugin protocols (e.g., oil discrimination, thermal stability, oil vs chips, HSI).
- **Validation strategy**: stratified, batch-aware, nested (if enabled).
- **Preprocessing summary**: shows baseline, smoothing, normalization, harmonization settings pulled from the selected protocol (read-only; advanced users can override if enabled).
- **HSI options**: segmentation method, clusters (if protocol supports HSI).
- **Minimal panel / clustering toggles**: enable/disable in RQ analysis if the protocol allows.

## Applying an existing model
- Use the “Apply existing model” button to load a frozen model and score new data. The GUI applies the same preprocessing/feature definitions and shows predictions/confusion matrix.

## Errors and validation
- The **Validate** button runs protocol/dataset checks. Blocking errors are shown in a dialog; warnings appear in the Overview/QC pane and run log.
- Common checks: missing columns, insufficient class counts, incompatible protocol/library versions.

## Where outputs go
- Each run creates a timestamped folder with `report.txt/html`, `figures/`, `tables/`, `metadata.json`, `index.json`, `run.log`, and optionally `models/`. The cockpit shows a link to open it.

## Advanced behaviors
- **Validation dialog:** blocks on missing columns/low class counts/incompatible versions.  
- **Cancel/timeout:** long runs execute in background threads; cancel stops between steps.  
- **Project (multi-dataset):** load multiple files, assign roles (instrument/matrix/batch), and run harmonized protocols; run history groups by project.  
- **HSI views:** HSI tab shows mean/label maps and ROI summaries when protocols include `hsi_segment/hsi_roi_to_1d`.

See also: [first-steps_gui.md](../01-getting-started/first-steps_gui.md) and [cookbook_troubleshooting.md](../03-cookbook/cookbook_troubleshooting.md).
