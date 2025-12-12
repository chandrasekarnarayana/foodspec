# First Steps (GUI)

- Launch cockpit: `python scripts/foodspec_protocol_cockpit.py`
- Launch wizard (guided): `python scripts/foodspec_protocol_wizard.py`
- Use presets: “Typical edible oil discrimination”, “Thermal stability tracking”, “Oil vs chips matrix comparison”.
- Keep defaults (5-fold CV, reference normalization) for a safe start.
- See docs/quickstart_protocol.md for expected columns and troubleshooting tips.
# First steps (GUI)

Start with the GUI for a gentle, visual introduction to FoodSpec protocols and outputs.

## Why start with the GUI?
- Less configuration: protocols and defaults are pre-wired.
- Visual feedback: validation warnings, plots, and run history are visible.

## Step-by-step
1) **Launch the cockpit or wizard**  
   ```bash
   foodspec-gui
   ```  
   - Cockpit: direct access to all tabs.  
   - Wizard: guided steps (Load → Validate → Configure → Run → Review).

2) **Load the example oil dataset**  
   - File: `examples/data/oils.csv` (or the bundled HDF5).  
   - In the **Data & Mapping** panel, browse to the file. Columns (oil_type, heating_stage, replicate) should auto-map; adjust if needed.

3) **Select the protocol**  
   - Protocol drop-down: choose **“Edible oil discrimination (basic)”** (`examples/protocols/oil_basic.yaml`).  
   - The **Run Plan** sidebar shows steps (preprocess → harmonize → QC → RQ → output) and key parameters (baseline, normalization, validation strategy).

4) **Validate and run**  
   - Click **Validate** to check required columns and class counts; blocking errors appear in a dialog, warnings in the QC/validation pane.  
   - Click **Run**. Progress appears in the status/log panel; cancel is available.

5) **Review outputs in the GUI**  
   - Tabs:  
     - **Overview**: executive summary, QC/validation warnings.  
     - **Discriminative**: confusion matrix/balanced accuracy, top ratios, minimal marker panel.  
     - **Stability**: CV/MAD tables and plots.  
     - **Heating Trends**: slopes/Spearman ρ, monotonicity (if heating_stage present).  
     - **Oil vs Chips**: divergence metrics (if matrix data present).  
     - **HSI**: segmentation/ROI views (if HSI data).  
   - Run history: shows recent runs with links to run folders.

## Where the files go (end-to-end example)
- The UI shows a link to the timestamped run folder. Inside you’ll see:  
  `report.txt/html`, `tables/`, `figures/`, `metadata.json`, `index.json`, `run.log`, and optionally `models/` (if a frozen model was created).
- First figures to open: confusion matrix (discriminative tab), top discriminative ratios, and stability plot.

## What you should see
- A balanced-accuracy/confusion matrix showing oil separation.
- A ranked list of discriminative peaks/ratios and a minimal marker panel with accuracy.
- Stability (CV/MAD) for key ratios; validation strategy (e.g., batch-aware CV) in the QC pane.

If something fails, the GUI shows a plain-language message box with next steps (e.g., check expected columns, install GUI extras). See [faq_basic.md](faq_basic.md) for common issues.
