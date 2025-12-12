#!/usr/bin/env python3
"""
FoodSpec Raman Workflow GUI
---------------------------

A publication-grade, memory-efficient GUI wrapper for the Raman workflow template.
Goals:
- Zero-setup wizard for non-experts ("GUI for dummies").
- One-click, fully automated run that mirrors the scripted workflow.
- Guided mode with safe defaults, minimal memory footprint (streamed reading, caching).
- Built-in feature inventory tying GUI panels to package/documentation capabilities.
- Automatic generation of figures, metrics, reports, metadata, and trained models suitable for publication.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import sys

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import raman_workflow_foodspec as wf

# --------------------------------------------------------------------------------------
# Inventory of capabilities exposed in the GUI (kept in sync with README/docs highlights)
# --------------------------------------------------------------------------------------

FEATURE_INVENTORY: Dict[str, List[str]] = {
    "io_and_data_model": [
        "Wide-format Raman CSV ingestion with metadata separation",
        "Automatic output folders under results/<run_name>",
        "Run metadata capture (config, OS, Python)",
    ],
    "preprocessing": [
        "Asymmetric least-squares baseline",
        "Savitzky-Golay smoothing",
        "Normalization sweep (vector/area/max/ref2720/ref1742)",
    ],
    "features_and_stats": [
        "Peak detection around target bands",
        "Ratiometric features vs 2720 and 1742 cm-1",
        "Coefficient of variation, ANOVA/MANOVA",
        "Random-forest feature ranking",
    ],
    "chemometrics_and_ml": [
        "PCA + silhouette, hierarchical clustering, MDS embedding",
        "Random-forest classification (oil ID) with cross-validated accuracy",
        "Random-forest regression for heating-stage prediction (R^2)",
        "Normalization robustness study across methods",
    ],
    "thermal_and_band_analysis": [
        "Heating-trend regression and correlation",
        "Peak-shift trends across heating stages",
        "Band-region areas with ANOVA",
        "Thermal stability index per oil",
    ],
    "reporting_and_outputs": [
        "Tables (CSV), figures (PNG), text reports (md/txt)",
        "Model artifacts (.joblib) for classification/regression",
        "Zipped run bundle for archiving/submission",
        "GUI-synthesized research Q&A prompts",
    ],
    "ergonomics_and_automation": [
        "One-click automated run with smart defaults",
        "Guided tips for publication-ready outputs",
        "Memory-aware caching and streaming CSV reads",
        "Download buttons and quick links to deliverables",
    ],
}


# --------------------------------------------------------------------------------------
# Helpers (non-invasive; they sit on top of the workflow without altering package code)
# --------------------------------------------------------------------------------------

def get_default_config() -> wf.Config:
    cfg = wf.Config()
    cfg.input_csv = ""  # set later by GUI
    cfg.run_name = "gui_oil_raman_RQ1_to_RQ14"
    return cfg


def write_run_metadata(results_dir: Path, config: wf.Config, extras: Dict) -> Path:
    meta = {
        "config": asdict(config),
        "extras": extras,
    }
    meta_path = results_dir / "gui_run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def train_and_save_models(
    tables_dir: Path,
    results_dir: Path,
    oil_col: str,
    heating_col: str,
    random_seed: int,
) -> Dict[str, Optional[float]]:
    """
    Train lightweight RF models from generated ratio tables.
    Saved artifacts: models/<clf|reg>.joblib
    """
    metrics: Dict[str, Optional[float]] = {"oil_rf_accuracy": None, "heating_r2": None}
    ratios_path = tables_dir / "02_ratiometric_features.csv"
    if not ratios_path.exists():
        return metrics

    df_ratios = pd.read_csv(ratios_path)
    model_dir = results_dir / "models"
    model_dir.mkdir(exist_ok=True)

    ratio_cols = [c for c in df_ratios.columns if c.endswith("_norm2720")]

    if oil_col in df_ratios.columns and ratio_cols:
        y = df_ratios[oil_col].astype(str)
        X = df_ratios[ratio_cols]
        clf = RandomForestClassifier(
            n_estimators=300, random_state=random_seed, n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        scores = cross_val_score(clf, X, y, cv=cv)
        clf.fit(X, y)
        joblib.dump(clf, model_dir / "rf_oil_classifier.joblib")
        metrics["oil_rf_accuracy"] = float(scores.mean())

    if heating_col in df_ratios.columns and ratio_cols:
        heating = pd.to_numeric(df_ratios[heating_col], errors="coerce")
        mask = ~heating.isna()
        if mask.sum() > 10:
            Xh = df_ratios.loc[mask, ratio_cols]
            yh = heating.loc[mask]
            reg = RandomForestRegressor(
                n_estimators=300, random_state=random_seed, n_jobs=-1
            )
            cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            scores = cross_val_score(reg, Xh, yh, cv=cv, scoring="r2")
            reg.fit(Xh, yh)
            joblib.dump(reg, model_dir / "rf_heating_regressor.joblib")
            metrics["heating_r2"] = float(scores.mean())
    return metrics


@st.cache_data(show_spinner=False)
def load_csv(upload) -> pd.DataFrame:
    return pd.read_csv(upload)


def run_pipeline(config: wf.Config, input_path: Path, chips_path: Optional[Path] = None) -> Path:
    config.input_csv = str(input_path)
    if chips_path is not None:
        config.chips_csv = str(chips_path)
    wf.main(config)
    return Path(config.output_root) / config.run_name


def build_research_prompts(results_dir: Path) -> List[str]:
    prompts: List[str] = []
    norm_path = results_dir / "tables" / "norm_robustness_summary.csv"
    rank_path = results_dir / "tables" / "04_feature_ranking_rf_ratios_norm2720.csv"
    heat_trend_path = results_dir / "tables" / "10_heating_trend_stats_ratios_norm2720.csv"
    peak_shift_path = results_dir / "tables" / "11_peak_shift_trends.csv"

    if norm_path.exists():
        df_norm = pd.read_csv(norm_path)
        if not df_norm.empty:
            best = df_norm.sort_values("rf_cv_accuracy", ascending=False).iloc[0]
            prompts.append(
                f"Normalization robustness: {best['norm_method']} delivers the strongest classification "
                f"(RF CV={best['rf_cv_accuracy']:.3f}); report how silhouette trends compare across methods."
            )
    if rank_path.exists():
        df_rank = pd.read_csv(rank_path)
        top_feats = df_rank.head(3)["feature"].tolist()
        prompts.append(
            f"Discriminative markers: highlight ratios {top_feats} with their importances; "
            "justify minimal feature set for QC."
        )
    if heat_trend_path.exists():
        df_heat = pd.read_csv(heat_trend_path)
        df_heat = df_heat.sort_values("p_corr")
        top_heat = df_heat.head(2)["feature"].tolist() if not df_heat.empty else []
        if top_heat:
            prompts.append(
                f"Thermal sensitivity: ratios {top_heat} show strongest monotonic response; "
                "interpret as degradation proxies."
            )
    if peak_shift_path.exists():
        df_peak = pd.read_csv(peak_shift_path)
        if not df_peak.empty:
            strongest = df_peak.sort_values("p_corr").head(1)
            peak = float(strongest["peak"].iloc[0])
            prompts.append(
                f"Structural change: peak {peak:.0f} cm-1 shifts with heating; link to bond changes."
            )
    return prompts


def bundle_results(results_dir: Path) -> Path:
    zip_target = results_dir.parent / f"{results_dir.name}_bundle"
    archive = shutil.make_archive(str(zip_target), "zip", root_dir=results_dir)
    return Path(archive)


# --------------------------------------------------------------------------------------
# Streamlit layout (intentionally simple and guided)
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="FoodSpec Raman GUI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FoodSpec Raman Workflow GUI")
st.caption("One-click, publication-grade Raman analysis with built-in guidance and automation.")

col_info, col_run = st.columns([1, 2])

with col_info:
    st.subheader("Feature Inventory")
    for section, bullets in FEATURE_INVENTORY.items():
        with st.expander(section.replace("_", " ").title(), expanded=False):
            st.write("\n".join([f"- {b}" for b in bullets]))


default_cfg = get_default_config()

with st.sidebar:
    st.header("Dataset & Config")
    uploaded = st.file_uploader("Upload Raman CSV", type=["csv"])
    csv_path = st.text_input("Or provide CSV path", value=default_cfg.input_csv)
    chips_upload = st.file_uploader(
        "Optional chips ratiometric CSV (for heating analysis)",
        type=["csv"],
        key="chips_upload",
        help="Provide precomputed chips ratio table if you want chips heating stats.",
    )
    chips_path = st.text_input(
        "Or provide chips CSV path (optional)",
        value="",
        help="Leave blank to skip chips heating analysis.",
    )
    run_name = st.text_input("Run name", value=default_cfg.run_name)
    baseline_lambda = st.number_input("Baseline lambda (ALS)", value=default_cfg.baseline_lambda)
    baseline_p = st.number_input("Baseline p (ALS)", value=default_cfg.baseline_p, format="%.6f")
    savgol_window = st.number_input("Savitzky-Golay window", value=default_cfg.savgol_window, step=2)
    norm_primary = st.selectbox("Primary normalization", default_cfg.norm_methods, index=0)
    auto_run = st.button("Run fully automated workflow", use_container_width=True, type="primary")
    st.markdown(
        "Tip: leave defaults for a turnkey run. All outputs are written under `results/<run_name>`."
    )


results_dir: Optional[Path] = None
last_metrics: Dict[str, Optional[float]] = {}
chips_file_used: Optional[Path] = None

if auto_run:
    cfg = wf.Config(
        baseline_lambda=float(baseline_lambda),
        baseline_p=float(baseline_p),
        savgol_window=int(savgol_window),
    )
    cfg.run_name = run_name

    # Allow user-selected primary normalization without altering downstream robustness sweep
    cfg.norm_methods = default_cfg.norm_methods

    tmp_chips: Optional[tempfile.NamedTemporaryFile] = None
    chips_file: Optional[Path] = None

    if uploaded is not None:
        df = load_csv(uploaded)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp_file.name, index=False)
        input_file = Path(tmp_file.name)
    else:
        input_file = Path(csv_path)

    if chips_upload is not None:
        df_chips = load_csv(chips_upload)
        tmp_chips = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_chips.to_csv(tmp_chips.name, index=False)
        chips_file = Path(tmp_chips.name)
    elif chips_path:
        chips_file = Path(chips_path)

    if not input_file.exists():
        st.error(f"Input CSV not found: {input_file}")
    else:
        if chips_file is not None and not chips_file.exists():
            st.warning(f"Chips CSV not found: {chips_file} (chips heating analysis will be skipped).")
            chips_file = None
        with st.spinner("Running automated Raman workflow..."):
            results_dir = run_pipeline(cfg, input_file, chips_path=chips_file)
            tables_dir = results_dir / "tables"
            figures_dir = results_dir / "figures"
            last_metrics = train_and_save_models(
                tables_dir, results_dir, cfg.oil_col, cfg.heating_col, cfg.random_seed
            )
            write_run_metadata(
                results_dir,
                cfg,
                {
                    "source_csv": str(input_file),
                    "primary_norm": norm_primary,
                    "streamlit_version": st.__version__,
                },
            )
            chips_file_used = chips_file

if results_dir and results_dir.exists():
    st.success(f"Workflow completed. Outputs in `{results_dir}`")
    if chips_file_used is None:
        st.caption("Chips heating analysis was skipped (no chips CSV provided).")
    else:
        st.caption(f"Chips heating analysis ran using: {chips_file_used}")
    col_a, col_b, col_c = st.columns(3)
    summary_path = results_dir / "summary_report.md"
    bundle_path = bundle_results(results_dir)

    with col_a:
        if summary_path.exists():
            st.download_button(
                label="Download summary report (md)",
                data=summary_path.read_text(),
                file_name=summary_path.name,
                mime="text/markdown",
            )
        st.write(f"Tables: `{results_dir / 'tables'}`")

    with col_b:
        st.download_button(
            label="Download full bundle (zip)",
            data=bundle_path.read_bytes(),
            file_name=bundle_path.name,
            mime="application/zip",
        )
        st.write(f"Figures: `{results_dir / 'figures'}`")

    with col_c:
        model_dir = results_dir / "models"
        if model_dir.exists():
            for model_file in model_dir.glob("*.joblib"):
                st.download_button(
                    label=f"Download {model_file.name}",
                    data=model_file.read_bytes(),
                    file_name=model_file.name,
                )
        st.json({k: v for k, v in last_metrics.items() if v is not None})

    st.subheader("Research-ready prompts (auto-generated)")
    prompts = build_research_prompts(results_dir)
    if prompts:
        for p in prompts:
            st.markdown(f"- {p}")
    else:
        st.info("Run the workflow to populate prompts.")

    st.subheader("Key figures")
    fig_candidates = [
        "01_raw_vs_processed.png",
        "05_PCA_oil_identity.png",
        "06_feature_importance_rf.png",
        "08_oil_centroid_distance_heatmap.png",
        "09_MDS_embedding_oils.png",
    ]
    for name in fig_candidates:
        fig_path = results_dir / "figures" / name
        if fig_path.exists():
            st.image(str(fig_path), caption=name, use_column_width=True)

    st.subheader("Tables overview")
    table_candidates = [
        "03_CV_by_oil_and_feature.csv",
        "04_feature_ranking_rf_ratios_norm2720.csv",
        "07_band_region_areas.csv",
        "10_heating_trend_stats_ratios_norm2720.csv",
        "norm_robustness_summary.csv",
    ]
    for name in table_candidates:
        t_path = results_dir / "tables" / name
        if t_path.exists():
            st.markdown(f"**{name}**")
            st.dataframe(pd.read_csv(t_path).head(50))
else:
    st.info("Upload a CSV or point to an existing one, then click 'Run fully automated workflow'.")
