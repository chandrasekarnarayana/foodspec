#!/usr/bin/env python3
"""
Unified FoodSpec GUI
--------------------

Streamlit front-end that wraps the full FoodSpec package:
- Data ingestion: example data or wide-format CSV -> FoodSpectrumSet
- Workflows: oil authentication, heating degradation, QC/novelty, Raman RQ1-RQ14 template
- Outputs: metrics tables inline, downloadable artifacts where available
- Feature inventory: maps GUI panels to package capabilities and docs

Design principles:
- Non-invasive: only orchestrates existing APIs (apps, chemometrics, workflows); core package untouched.
- Memory-aware: small DataFrame previews, cached CSV parsing, temporary files cleaned up.
- Dummy-friendly: defaults pre-filled, minimal required inputs, inline tips.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from foodspec.apps.heating import run_heating_degradation_analysis
from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.apps.qc import run_qc_workflow
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.data.loader import load_example_oils
from foodspec.validation import validate_spectrum_set
from scripts import raman_workflow_foodspec as raman_template

# --------------------------------------------------------------------------------------
# Inventory (aligned with README/docs)
# --------------------------------------------------------------------------------------

FEATURE_INVENTORY: Dict[str, List[str]] = {
    "Data & IO": [
        "Wide-format CSV ingestion to FoodSpectrumSet",
        "Synthetic example oils dataset",
        "Run metadata capture via outputs written by workflows",
    ],
    "Preprocessing": [
        "Baseline (ALS), Savitzky-Golay smoothing",
        "Vector/area/max/reference normalizations (via Raman template)",
        "Cropping to fingerprint region where relevant",
    ],
    "Features & Stats": [
        "Peak and ratio extraction (oils/heating)",
        "Coefficient of variation, ANOVA/MANOVA (Raman template)",
        "Random-forest feature ranking",
    ],
    "Chemometrics & ML": [
        "RF classifier/regressor with CV (oil ID, heating stage)",
        "QC/novelty detection (OneClass SVM / Isolation Forest)",
        "PCA/cluster/embedding via Raman RQ1-RQ14 template",
    ],
    "Thermal & Band Analysis": [
        "Heating trend regression and peak shifts",
        "Band-region areas and ANOVA (Raman template)",
        "Thermal stability index (Raman template)",
    ],
    "Reporting & Outputs": [
        "Tables/figures/report from Raman template under results/<run_name>",
        "Inline metrics tables and confusion matrices for workflows",
        "Temporary CSV export to run the Raman template from GUI-loaded data",
    ],
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def parse_meta_list(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@st.cache_data(show_spinner=False)
def load_csv_as_spectrum_set(
    file_bytes: bytes,
    meta_cols: Sequence[str],
    modality: str,
) -> FoodSpectrumSet:
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    meta_cols = [c for c in meta_cols if c in df.columns]
    spec_cols = [c for c in df.columns if c not in meta_cols]
    wn = np.array([float(c) for c in spec_cols], dtype=float)
    X = df[spec_cols].to_numpy(dtype=float)
    metadata = df[meta_cols].reset_index(drop=True)
    return FoodSpectrumSet(x=X, wavenumbers=wn, metadata=metadata, modality=modality)


def spectrum_set_to_wide_csv(fs: FoodSpectrumSet, path: Path) -> None:
    """Export FoodSpectrumSet to a wide CSV compatible with the Raman template."""
    df_meta = fs.metadata.reset_index(drop=True)
    df_spec = pd.DataFrame(fs.x, columns=[f"{wn:.4f}" for wn in fs.wavenumbers])
    df_out = pd.concat([df_meta, df_spec], axis=1)
    df_out.to_csv(path, index=False)


def ensure_valid(fs: FoodSpectrumSet) -> FoodSpectrumSet:
    validate_spectrum_set(fs)
    return fs


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="FoodSpec GUI (Full Package)", layout="wide")
st.title("FoodSpec GUI (Full Package)")
st.caption(
    "Covers oil authentication, heating, QC/novelty, and the Raman RQ1-RQ14 template. Defaults are set for non-experts."
)

with st.sidebar:
    st.header("Data source")
    data_mode = st.radio("Choose dataset source", ["Example oils", "Upload CSV (wide format)"])
    modality = st.selectbox("Modality", ["raman", "ftir", "nir"], index=0)
    meta_text = st.text_input("Metadata columns (comma-separated)", value="oil_type,heating_time")
    meta_cols = parse_meta_list(meta_text)
    st.markdown("Metadata columns are pulled from the CSV; all other columns are treated as wavenumbers.")

fs: Optional[FoodSpectrumSet] = None
source_name = ""

if data_mode == "Example oils":
    fs = load_example_oils()
    source_name = "example_oils (synthetic)"
else:
    uploaded = st.file_uploader("Upload wide-format CSV", type=["csv"])
    if uploaded is not None:
        try:
            fs = load_csv_as_spectrum_set(uploaded.getvalue(), meta_cols=meta_cols, modality=modality)
            source_name = uploaded.name
        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")

if fs is not None:
    ensure_valid(fs)
    st.success(f"Loaded dataset '{source_name}' with {len(fs)} spectra, modality={fs.modality}")
    st.write("Metadata preview")
    st.dataframe(fs.metadata.head(20))
    st.write(f"Wavenumbers: {len(fs.wavenumbers)} points ({fs.wavenumbers.min():.1f}-{fs.wavenumbers.max():.1f})")
else:
    st.info("Load data to enable workflows.")

st.subheader("Feature Inventory")
cols_inv = st.columns(3)
keys = list(FEATURE_INVENTORY.keys())
for i, key in enumerate(keys):
    with cols_inv[i % 3]:
        st.markdown(f"**{key}**")
        st.write("\n".join([f"- {item}" for item in FEATURE_INVENTORY[key]]))

st.subheader("Workflows")
tab_oil, tab_heat, tab_qc, tab_raman = st.tabs(
    ["Oil authentication", "Heating degradation", "QC / novelty", "Raman RQ1-RQ14 template"]
)

with tab_oil:
    st.markdown("Runs `foodspec.apps.oils.run_oil_authentication_quickstart` with minimal inputs.")
    label_col = st.text_input("Label column (oil type)", value="oil_type")
    if st.button("Run oil authentication", type="primary", disabled=fs is None):
        if fs is None:
            st.error("Load data first.")
        elif label_col not in fs.metadata.columns:
            st.error(f"Metadata column '{label_col}' not found.")
        else:
            with st.spinner("Running oil authentication..."):
                res = run_oil_authentication_quickstart(fs, label_column=label_col, cv_splits=3)
            st.success("Completed oil authentication.")
            st.markdown("**Cross-validated metrics (mean over folds)**")
            st.dataframe(res.cv_metrics)
            st.markdown("**Confusion matrix**")
            st.dataframe(pd.DataFrame(res.confusion_matrix, index=res.class_labels, columns=res.class_labels))
            if res.feature_importances is not None:
                st.markdown("**Top features (RF importance)**")
                st.dataframe(res.feature_importances.sort_values(ascending=False).head(20))

with tab_heat:
    st.markdown("Heating degradation analysis with peak ratios and trend models.")
    time_col = st.text_input("Heating time/stage column", value="heating_time")
    if st.button("Run heating analysis", type="primary", disabled=fs is None):
        if fs is None:
            st.error("Load data first.")
        elif time_col not in fs.metadata.columns:
            st.error(f"Metadata column '{time_col}' not found.")
        else:
            with st.spinner("Running heating analysis..."):
                res = run_heating_degradation_analysis(fs, time_column=time_col)
            st.success("Completed heating analysis.")
            st.markdown("**Ratio table (head)**")
            st.dataframe(res.ratio_table.head(30))
            st.markdown("**Trend slopes (per ratio)**")
            st.json({k: float(v.coef_[0]) if hasattr(v, "coef_") else None for k, v in res.trend_models.items()})
            if res.anova_table is not None:
                st.markdown("**ANOVA (if groups present)**")
                st.dataframe(res.anova_table)

with tab_qc:
    st.markdown("QC / novelty detection using OneClass SVM or Isolation Forest.")
    train_mask_col = st.text_input("Reference mask column (optional boolean)", value="")
    model_type = st.selectbox("Model type", ["oneclass_svm", "isolation_forest"])
    if st.button("Run QC / novelty", type="primary", disabled=fs is None):
        if fs is None:
            st.error("Load data first.")
        else:
            mask = None
            if train_mask_col:
                if train_mask_col not in fs.metadata.columns:
                    st.error(f"Mask column '{train_mask_col}' not found.")
                    mask = None
                else:
                    mask = fs.metadata[train_mask_col].astype(bool)
            if mask is not None or train_mask_col == "":
                with st.spinner("Running QC workflow..."):
                    res = run_qc_workflow(fs, train_mask=mask, model_type=model_type)
                st.success("Completed QC workflow.")
                st.markdown("**Scores (head)**")
                st.dataframe(res.scores.head(50))
                st.markdown("**Predicted labels count**")
                st.json(res.predicted_labels.value_counts().to_dict())
                st.markdown(f"Threshold: {res.threshold:.3f} (higher is normal={res.higher_score_is_more_normal})")

with tab_raman:
    st.markdown("Full RQ1-RQ14 Raman template (tables/figures/report under `results/<run_name>`).")
    run_name = st.text_input("Run name", value="gui_full_package_raman")
    oil_col = st.text_input("Oil label column (optional)", value="oil_type")
    heating_col = st.text_input("Heating column (optional)", value="heating_time")
    if st.button("Run Raman RQ1-RQ14 template", type="primary", disabled=fs is None):
        if fs is None:
            st.error("Load data first.")
        else:
            with tempfile.TemporaryDirectory() as td:
                tmp_csv = Path(td) / "raman_gui_tmp.csv"
                # Export current dataset to a compatible wide CSV
                spectrum_set_to_wide_csv(fs, tmp_csv)
                cfg = raman_template.Config(
                    input_csv=str(tmp_csv),
                    run_name=run_name,
                    oil_col=oil_col,
                    heating_col=heating_col,
                )
                with st.spinner("Running Raman template..."):
                    raman_template.main(cfg)
            st.success(f"Raman template finished. See results/{run_name}")
            st.markdown(
                f"Artifacts saved to `results/{run_name}` (figures, tables, summary_report.md, models if generated)."
            )

st.write("---")
st.caption(
    "The GUI is a thin orchestration layer; all computations are delegated to FoodSpec workflows to stay aligned with package behavior."
)
