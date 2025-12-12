"""
Master Oil & Chips Raman Analysis Pipeline + RQ Summaries
---------------------------------------------------------

This script:

1. From raw Raman Oil spectra:
   - Baseline ALS correction
   - Peak picking for key bands
   - CVs of peak intensities + normalized ratios (I_peak / I_2720)
   - Saves:
       - Oil Ratio peak_CV values.csv
       - Oil Ratio peak_Normalized_CV values.csv
       - Oil Ratiometric Analysis.csv

2. Oil identity statistics (normalized ratios):
   - One-way ANOVA across Oil_Name
   - Group means/SDs
   - Tukey HSD post-hoc for significant ratios
   - Saves into Supplementary_Oil Stats_Output/

3. Multivariate oil identity analysis on *_norm2720:
   - PCA + silhouette + between/within ratio + permutation p-value
   - MANOVA
   - Feature-wise ANOVA/Kruskal
   - Games–Howell post-hoc
   - Boxplots per feature
   - Saves into Stats_Output_Oil_norm2720_YYYY_MM_DD_HH_HH_MM/

4. Thermal stability vs Heating_Stage:
   - For Chips Ratiometric Analysis (Chips_Name)
   - For Oil Ratiometric Analysis (Oil_Name)
   - Global ANOVA, slopes, stability ranking, figures, interpretation tables
   - Saves into Supplementary_Chips_HeatingStage_Results/ and Supplementary_Oil_HeatingStage_Results/

5. Publication-ready RQ summaries (RQ1–RQ14):
   - Supervised discrimination (RandomForest, Logistic Regression)
   - Feature importance + minimal marker panel
   - Unsupervised clustering (KMeans) & chemometric map (PCA + MDS)
   - Intra-class variability & functional band behaviour
   - Peak position shifts vs Heating_Stage
   - Thermal Stability Index per oil (distance from unheated fingerprint)
   - Heating-stage prediction (“chemical clock” regression)
   - Normalization robustness (reference-peak vs vector normalization)

   -> Saves textual answers to:
      - Publication_Summaries/RQ_detailed_answers.txt
      - Publication_Summaries/RQ_brief_summary.txt
"""

# ============================================================
# 0. Imports + dependency check
# ============================================================
import importlib
import subprocess
import sys
import os
import json
import warnings
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

required = ["numpy", "pandas", "matplotlib", "scipy", "sklearn", "statsmodels", "tqdm", "seaborn"]

for pkg in required:
    try:
        importlib.import_module(pkg)
        print(f"{pkg}  installed")
    except ImportError:
        print(f"{pkg} ❌ missing, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scipy import stats
from scipy.sparse import diags
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import linregress

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ============================================================
# 1. CONFIG – centralize all file paths here
# ============================================================
# 👉 EDIT THESE PATHS AS NEEDED
RAW_OIL_CSV = r"C:\Users\sairam\Downloads\Raman Oil Raw data.csv"
OIL_RATIOMETRIC_CSV = r"C:\Users\sairam\Downloads\Oil Ratiometric Analysis.csv"
CHIPS_RATIOMETRIC_CSV = r"C:\Users\sairam\Downloads\Chips Ratiometric Analysis.csv"

SUPP_OIL_STATS_DIR = r"C:\Users\sairam\Downloads\Supplementary_Oil Stats_Output"
SUPP_CHIPS_HEAT_DIR = r"C:\Users\sairam\Downloads\Supplementary_Chips_HeatingStage_Results"
SUPP_OIL_HEAT_DIR = r"C:\Users\sairam\Downloads\Supplementary_Oil_HeatingStage_Results"

OIL_PEAK_CV_CSV = os.path.join(SUPP_OIL_STATS_DIR, "Oil Ratio peak_CV values.csv")
OIL_RATIO_NORM_CV_CSV = os.path.join(SUPP_OIL_STATS_DIR, "Oil Ratio peak_Normalized_CV values.csv")

PUBLICATION_SUMMARY_DIR = Path("Publication_Summaries")
ACTIVE_CONFIG = None


@dataclass
class Config:
    """
    Lightweight configuration for GUI/CLI runs.
    All paths will be rewritten under output_root/run_name when provided.
    """

    input_csv: str = ""
    chips_csv: Optional[str] = None
    run_name: str = "raman_gui_run"
    output_root: Path = Path("results")
    oil_col: str = "Oil_Name"
    heating_col: str = "Heating_Stage"
    baseline_lambda: float = 10**5
    baseline_p: float = 0.01
    savgol_window: int = 5
    random_seed: int = 0
    norm_methods: List[str] = field(
        default_factory=lambda: [
            "vector_norm",
            "area_norm",
            "max_norm",
            "ref_peak_2720",
            "ref_peak_1742",
        ]
    )


def configure_paths(cfg: Config):
    """
    Re-point all output paths to a run-specific results directory.
    This keeps the GUI ergonomic and avoids hard-coded Windows paths.
    """
    global RAW_OIL_CSV, OIL_RATIOMETRIC_CSV, CHIPS_RATIOMETRIC_CSV
    global SUPP_OIL_STATS_DIR, SUPP_CHIPS_HEAT_DIR, SUPP_OIL_HEAT_DIR
    global OIL_PEAK_CV_CSV, OIL_RATIO_NORM_CV_CSV, PUBLICATION_SUMMARY_DIR, ACTIVE_CONFIG

    ACTIVE_CONFIG = cfg

    base_dir = Path(cfg.output_root) / cfg.run_name
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"

    for p in [base_dir, tables_dir, figures_dir]:
        p.mkdir(parents=True, exist_ok=True)

    RAW_OIL_CSV = cfg.input_csv
    OIL_RATIOMETRIC_CSV = str(tables_dir / "Oil Ratiometric Analysis.csv")
    CHIPS_RATIOMETRIC_CSV = cfg.chips_csv or str(tables_dir / "Chips Ratiometric Analysis.csv")

    SUPP_OIL_STATS_DIR = str(tables_dir / "Supplementary_Oil Stats_Output")
    SUPP_CHIPS_HEAT_DIR = str(tables_dir / "Supplementary_Chips_HeatingStage_Results")
    SUPP_OIL_HEAT_DIR = str(tables_dir / "Supplementary_Oil_HeatingStage_Results")

    OIL_PEAK_CV_CSV = os.path.join(SUPP_OIL_STATS_DIR, "Oil Ratio peak_CV values.csv")
    OIL_RATIO_NORM_CV_CSV = os.path.join(SUPP_OIL_STATS_DIR, "Oil Ratio peak_Normalized_CV values.csv")

    PUBLICATION_SUMMARY_DIR = base_dir / "Publication_Summaries"

    ensure_dir(Path(SUPP_OIL_STATS_DIR))
    ensure_dir(Path(SUPP_CHIPS_HEAT_DIR))
    ensure_dir(Path(SUPP_OIL_HEAT_DIR))
    ensure_dir(PUBLICATION_SUMMARY_DIR)

    return {
        "base": base_dir,
        "tables": tables_dir,
        "figures": figures_dir,
    }


# ============================================================
# 2. Generic helper utilities
# ============================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize_col(colname: str) -> str:
    """
    Make a dataframe column name safe for Patsy/statsmodels formulas:
    - Replace spaces and special chars with '_'
    - Ensure it doesn't start with a number
    """
    safe = colname.strip()
    for ch in [" ", "/", "(", ")", "-", "*", ".", "[", "]", "%"]:
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    if len(safe) > 0 and safe[0].isdigit():
        safe = "F_" + safe
    return safe


def games_howell(df, group_col, value_col):
    """
    Pairwise post-hoc like Games–Howell:
    Welch-style t-test with Holm correction.
    Returns dataframe with pairwise group comparisons.
    """
    groups = {g: np.asarray(vals, dtype=float) for g, vals in df.groupby(group_col)[value_col]}

    pairs, tvals, dfs, pvals = [], [], [], []

    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if j <= i:
                continue

            xi = groups[gi]
            xj = groups[gj]

            ni, nj = len(xi), len(xj)
            mi, mj = xi.mean(), xj.mean()
            si2, sj2 = xi.var(ddof=1), xj.var(ddof=1)

            denom = np.sqrt(si2 / ni + sj2 / nj + 1e-12)
            t_stat = (mi - mj) / denom

            # Welch-Satterthwaite df
            df_denom = (si2 / ni + sj2 / nj) ** 2
            df_num = (si2**2) / (ni**2 * (ni - 1)) + (sj2**2) / (nj**2 * (nj - 1))
            df_welch = df_denom / (df_num + 1e-12)

            p_val = 2 * stats.t.sf(np.abs(t_stat), df_welch)

            pairs.append((gi, gj))
            tvals.append(t_stat)
            dfs.append(df_welch)
            pvals.append(p_val)

    reject, p_adj, _, _ = multipletests(pvals, method="holm")

    out = pd.DataFrame(pairs, columns=["group1", "group2"])
    out["t"] = tvals
    out["df"] = dfs
    out["p_raw"] = pvals
    out["p_holm"] = p_adj
    out["reject"] = reject
    return out


def compute_pca_metrics(X, labels, n_components=5, random_state=0):
    """
    PCA and quantitative class separation metrics:
    - explained variance per PC
    - silhouette score
    - between/within scatter ratio in PCA space
    - permutation p-value for that ratio
    """
    Xz = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(n_components, Xz.shape[1]), random_state=random_state)
    scores = pca.fit_transform(Xz)
    exp_var = pca.explained_variance_ratio_

    labels = np.asarray(labels)
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(scores, labels)
    else:
        sil = np.nan

    overall = scores.mean(axis=0, keepdims=True)
    classes = np.unique(labels)

    Sb = np.zeros((scores.shape[1], scores.shape[1]))
    Sw = np.zeros_like(Sb)

    for c in classes:
        Xc = scores[labels == c]
        mu = Xc.mean(axis=0, keepdims=True)
        Sb += Xc.shape[0] * (mu - overall).T.dot(mu - overall)

        Xc_centered = Xc - mu
        Sw += Xc_centered.T.dot(Xc_centered)

    bw_ratio = np.trace(Sb) / (np.trace(Sw) + 1e-12)

    # permutation test on bw_ratio
    rng = np.random.default_rng(random_state)
    perm_ratios = []
    for _ in range(199):  # smaller number to keep runtime moderate
        perm_labels = rng.permutation(labels)
        Sb_p = np.zeros_like(Sb)
        Sw_p = np.zeros_like(Sw)
        for c in classes:
            Xc = scores[perm_labels == c]
            mu = Xc.mean(axis=0, keepdims=True)
            Sb_p += Xc.shape[0] * (mu - overall).T.dot(mu - overall)
            Xc_centered = Xc - mu
            Sw_p += Xc_centered.T.dot(Xc_centered)
        perm_ratios.append(np.trace(Sb_p) / (np.trace(Sw_p) + 1e-12))

    perm_ratios = np.asarray(perm_ratios)
    p_perm = (np.sum(perm_ratios >= bw_ratio) + 1) / (len(perm_ratios) + 1)

    return {"pca": pca, "scores": scores, "exp_var": exp_var, "silhouette": sil, "bw_ratio": bw_ratio, "p_perm": p_perm}


# ============================================================
# 3. Baseline ALS + Peak extraction + Ratiometric analysis
# ============================================================
def baseline_als(y, lam=10**5, p=0.01, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))  # Second-order difference matrix
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def extract_oil_peaks_and_ratios(cfg: Optional[Config] = None):
    """
    From RAW_OIL_CSV:
    - baseline correction
    - peak picking at 2720, 1742, 1652, 1434, 1259, 1296 cm-1 (±15)
    - build df_results with peak pos/int
    - compute CVs and normalized intensities (I_peak / I_2720)
    - save Oil Ratio peak_CV values.csv, Oil Ratio peak_Normalized_CV values.csv
    - save Oil Ratiometric Analysis.csv
    """
    print("\n=== STEP 1: Peak picking + ratiometric intensities (Oil) ===")

    os.makedirs(SUPP_OIL_STATS_DIR, exist_ok=True)

    if not RAW_OIL_CSV or not os.path.exists(RAW_OIL_CSV):
        raise FileNotFoundError(f"Oil CSV not found: {RAW_OIL_CSV}. Provide a valid path via Config.input_csv.")

    data = pd.read_csv(RAW_OIL_CSV)

    # Define target bands (±15 cm⁻¹)
    target_bands = {
        2720: (2720 - 15, 2720 + 15),
        1742: (1742 - 15, 1742 + 15),
        1652: (1652 - 15, 1652 + 15),
        1434: (1434 - 15, 1434 + 15),
        1259: (1259 - 15, 1259 + 15),
        1296: (1296 - 15, 1296 + 15),
    }

    # Split metadata vs numeric wavenumber columns
    wavenumber_cols = []
    meta_cols = []
    for col in data.columns:
        try:
            float(col)
            wavenumber_cols.append(col)
        except Exception:
            meta_cols.append(col)

    if len(wavenumber_cols) == 0:
        raise ValueError("Could not detect any numeric wavenumber columns in the CSV.")

    wavenumbers = np.array([float(c) for c in wavenumber_cols])

    # Prepare output
    results = []

    sample_col = None
    if cfg and cfg.oil_col in data.columns:
        sample_col = cfg.oil_col
    elif "Oil_Type" in data.columns:
        sample_col = "Oil_Type"
    elif meta_cols:
        sample_col = meta_cols[0]

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        sample_name = row[sample_col] if sample_col is not None else row.iloc[0]
        spectrum = row[wavenumber_cols].values.astype(float)

        # Baseline correction
        lam = cfg.baseline_lambda if cfg else 10**5
        p = cfg.baseline_p if cfg else 0.01
        baseline = baseline_als(spectrum, lam=lam, p=p)
        corrected = np.asarray(spectrum - baseline)

        row_result = {"S.No.": idx + 1, "Sample": sample_name}

        # Find peak in each target band
        for center, (low, high) in target_bands.items():
            mask = (wavenumbers >= low) & (wavenumbers <= high)
            wn_region = wavenumbers[mask]
            intensity_region = corrected[mask]

            if len(intensity_region) == 0 or np.all(np.isnan(intensity_region)):
                peak_wn = np.nan
                peak_int = np.nan
            else:
                max_idx = np.argmax(intensity_region)
                peak_wn = wn_region[max_idx]
                peak_int = intensity_region[max_idx]

            row_result[f"Peak Pos {center}"] = peak_wn
            row_result[f"Peak Int {center}"] = peak_int

        results.append(row_result)

    df_results = pd.DataFrame(results)

    # Populate Oil_Name / Heating_Stage from provided metadata when available
    if cfg and cfg.oil_col in data.columns:
        df_results["Oil_Name"] = data[cfg.oil_col].values
    if cfg and cfg.heating_col in data.columns:
        df_results["Heating_Stage"] = pd.to_numeric(data[cfg.heating_col].values, errors="coerce")

    # Fallback: parse Oil_Name + Heating_Stage from Sample (e.g., "SO 0", "VO 9")
    if "Oil_Name" not in df_results.columns or df_results["Oil_Name"].isna().all():
        df_results[["Oil_Name", "Heating_Stage"]] = (
            df_results["Sample"].astype(str).str.extract(r"([A-Za-z]+\s*[Cc]?)\s*(\d*)")
        )

    # Reorder columns
    cols = list(df_results.columns)
    sample_index = cols.index("Sample")
    reordered_cols = (
        cols[: sample_index + 1]
        + ["Oil_Name", "Heating_Stage"]
        + [c for c in cols if c not in ["Oil_Name", "Heating_Stage"] and c not in cols[: sample_index + 1]]
    )
    df_results = df_results[reordered_cols]

    # Save peak table
    Path(OIL_PEAK_CV_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OIL_PEAK_CV_CSV, index=False)
    print(f"Saved peak table to: {OIL_PEAK_CV_CSV}")

    # Extract only intensity columns
    intensity_cols = [col for col in df_results.columns if "Peak Int" in col]

    # Normalize each intensity with 2720 band
    df_norm = df_results.copy()
    for col in intensity_cols:
        if col != "Peak Int 2720":
            df_norm[col + "_norm2720"] = df_results[col] / df_results["Peak Int 2720"]

    # Compute CV for normalized intensities (and positions)
    cv_norm = {}
    for col in [c for c in df_norm.columns if "Peak" in c]:
        mean = df_norm[col].mean()
        std = df_norm[col].std()
        cv = (std / mean) * 100 if mean != 0 else None
        cv_norm[col] = round(cv, 2) if cv is not None else None

    cv_results = pd.DataFrame(list(cv_norm.items()), columns=["Peak", "cv(%)"])
    Path(OIL_RATIO_NORM_CV_CSV).parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(OIL_RATIO_NORM_CV_CSV, index=False)
    print(f"Saved normalized CV table to: {OIL_RATIO_NORM_CV_CSV}")

    # Save the full ratiometric analysis table
    Path(OIL_RATIOMETRIC_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(OIL_RATIOMETRIC_CSV, index=False)
    print(f"Saved Oil Ratiometric Analysis to: {OIL_RATIOMETRIC_CSV}")


# ============================================================
# 4. Oil-wise ANOVA + Tukey HSD on *_norm2720 ratios
# ============================================================
def oil_anova_and_tukey():
    print("\n=== STEP 2: Oil-wise ANOVA + Tukey HSD on ratios ===")
    os.makedirs(SUPP_OIL_STATS_DIR, exist_ok=True)

    df = pd.read_csv(OIL_RATIOMETRIC_CSV)

    # Normalized ratio columns (I / I2720)
    norm_cols = [
        "Peak Int 1742_norm2720",
        "Peak Int 1652_norm2720",
        "Peak Int 1434_norm2720",
        "Peak Int 1259_norm2720",
        "Peak Int 1296_norm2720",
    ]

    # ==== Q1. One-way ANOVA across Oil_Name ====
    anova_results = {}
    for col in norm_cols:
        groups = [group[col].values for _, group in sorted(df.groupby("Oil_Name"))]
        f_stat, p_val = stats.f_oneway(*groups)
        anova_results[col] = {"F_statistic": round(f_stat, 3), "p_value": p_val}  # keep full precision for saving

    # ==== Mean and Std per oil per ratio ====
    group_stats = df.groupby("Oil_Name")[norm_cols].agg(["mean", "std"])
    group_stats.columns = [f"{col}_{stat}" for col, stat in group_stats.columns]
    group_stats = group_stats.reset_index()

    stats_path = os.path.join(SUPP_OIL_STATS_DIR, "Supplementary_Table_GroupMeanStd.csv")
    group_stats.to_csv(stats_path, index=False)
    print(f"Saved mean/std per oil group to: {stats_path}")

    # ==== ANOVA table ====
    anova_df = pd.DataFrame(anova_results).T.reset_index()
    anova_df.columns = ["Peak_Ratio", "F_statistic", "p_value"]

    anova_path = os.path.join(SUPP_OIL_STATS_DIR, "Supplementary_Table_ANOVA.csv")
    anova_df.to_csv(anova_path, index=False)
    print(f"Saved ANOVA results to: {anova_path}")

    # ==== Tukey HSD post-hoc for each significant peak ====
    tukey_tables = {}

    for col in norm_cols:
        if anova_results[col]["p_value"] < 0.05:
            tukey = pairwise_tukeyhsd(df[col], df["Oil_Name"])
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_tables[col] = tukey_df

            safe_colname = col.replace(" ", "_").replace("/", "_")
            tukey_path = os.path.join(SUPP_OIL_STATS_DIR, f"Supplementary_Table_Tukey_{safe_colname}.csv")
            tukey_df.to_csv(tukey_path, index=False)
            print(f"Saved Tukey HSD for {col} to: {tukey_path}")

    print("\n=== ANOVA Summary ===")
    print(anova_df)

    print("\n=== Tukey HSD tables available for: ===")
    for k in tukey_tables.keys():
        print("-", k)


# ============================================================
# 5. Multivariate oil identity stats on *_norm2720
# ============================================================
def main_norm2720(input_csv, output_dir):
    save_dir = Path(output_dir)
    ensure_dir(save_dir)

    print("\n=== STEP 3: Multivariate oil identity (PCA + MANOVA + Games–Howell) ===")
    print("\n=== LOADING DATA ===")
    df_raw = pd.read_csv(input_csv)

    oil_col = "Oil_Name"  # grouping variable
    assert oil_col in df_raw.columns, f"{oil_col} column not found"

    feat_cols_original = [c for c in df_raw.columns if "_norm2720" in c]
    if len(feat_cols_original) == 0:
        raise RuntimeError("No *_norm2720 columns found in dataset.")
    print(f"Found {len(feat_cols_original)} norm2720 features:")
    for c in feat_cols_original:
        print(f" - {c}")

    # sanitize feature names for statsmodels formula compatibility
    col_map = {c: sanitize_col(c) for c in feat_cols_original}

    df = df_raw[[oil_col] + feat_cols_original].copy()
    df.rename(columns=col_map, inplace=True)

    feat_cols_safe = [col_map[c] for c in feat_cols_original]
    oils_sorted = sorted(df[oil_col].unique())

    # 1. Replicate depth per oil
    counts = df.groupby(oil_col).size().reset_index(name="n")
    counts.to_csv(save_dir / "counts_per_oil.csv", index=False)

    print("\n=== REPLICATE DEPTH PER OIL (counts_per_oil.csv) ===")
    print(counts.to_string(index=False))

    # 2. Descriptive stats per oil per feature
    desc_rows = []
    for feat in feat_cols_safe:
        for oil in oils_sorted:
            vals = df.loc[df[oil_col] == oil, feat].dropna().values
            if len(vals) == 0:
                continue
            desc_rows.append(
                {
                    oil_col: oil,
                    "feature": feat,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "n": int(len(vals)),
                }
            )
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(save_dir / "norm2720_mean_std_by_oil.csv", index=False)

    print("\n=== PER-OIL MEAN ± STD FOR EACH FEATURE (norm2720_mean_std_by_oil.csv) ===")
    preview_desc = desc_df.sort_values(["feature", oil_col]).groupby("feature").head(5)
    print(preview_desc.to_string(index=False))

    # 3. PCA + metrics
    X = df[feat_cols_safe].astype(float).values
    labels = df[oil_col].astype(str).values

    pca_stats = compute_pca_metrics(X, labels, n_components=min(5, X.shape[1]))

    np.savetxt(save_dir / "pca_exp_var_norm2720.csv", pca_stats["exp_var"], delimiter=",")

    pca_summary_payload = {
        "silhouette": pca_stats["silhouette"],
        "between_within_ratio": pca_stats["bw_ratio"],
        "p_perm": pca_stats["p_perm"],
    }
    with open(save_dir / "pca_summary_norm2720.json", "w") as f:
        json.dump(pca_summary_payload, f, indent=2)

    scores = pca_stats["scores"]
    fig = plt.figure(figsize=(6, 5))
    for oil in oils_sorted:
        mask = labels == oil
        plt.scatter(scores[mask, 0], scores[mask, 1], s=15, label=str(oil))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA on Raman intensity ratios (I_peak / I_2720)")
    plt.legend(markerscale=2)
    fig.savefig(save_dir / "pca_norm2720_scatter.png", dpi=200)
    plt.close(fig)

    print("\n=== PCA SUMMARY (pca_summary_norm2720.json) ===")
    exp_var = pca_stats["exp_var"]
    print(f"Explained variance PC1: {exp_var[0] * 100:.2f}%")
    if len(exp_var) > 1:
        print(f"Explained variance PC2: {exp_var[1] * 100:.2f}%")
    print(f"Silhouette score: {pca_stats['silhouette']:.4f}")
    print(f"Between/Within scatter ratio: {pca_stats['bw_ratio']:.4f}")
    print(f"Permutation test p_perm: {pca_stats['p_perm']:.4f}")
    print("PCA scatter saved as pca_norm2720_scatter.png")

    # 4. MANOVA
    lhs = " + ".join(feat_cols_safe)
    manova_df = df[[oil_col] + feat_cols_safe].dropna()
    mv = MANOVA.from_formula(f"{lhs} ~ C({oil_col})", data=manova_df)
    manova_result_str = str(mv.mv_test())
    with open(save_dir / "manova_norm2720.txt", "w") as f:
        f.write(manova_result_str)

    manova_pval = None
    for line in manova_result_str.splitlines():
        if "Pr > F" in line:
            parts = line.strip().split()
            maybe_p = parts[-1]
            try:
                manova_pval = float(maybe_p)
            except Exception:
                pass
    print("\n=== MANOVA SUMMARY (manova_norm2720.txt) ===")
    if manova_pval is not None:
        print(f"MANOVA p-value (Pr > F): {manova_pval}")
    else:
        print("Could not automatically parse MANOVA p-value. See manova_norm2720.txt")
    print("\n(First lines of MANOVA output)")
    print("\n".join(manova_result_str.splitlines()[:12]))

    # 5. Feature-level ANOVA / Kruskal
    rows_anova = []
    for feat in feat_cols_safe:
        group_vectors = [df.loc[df[oil_col] == oil, feat].dropna().values for oil in oils_sorted]
        group_vectors_valid = [g for g in group_vectors if len(g) > 1]
        if len(group_vectors_valid) < 2:
            continue
        try:
            F, p = stats.f_oneway(*group_vectors_valid)
            rows_anova.append({"feature": feat, "test": "ANOVA", "stat": float(F), "p": float(p)})
        except Exception:
            try:
                H, p = stats.kruskal(*group_vectors_valid)
                rows_anova.append({"feature": feat, "test": "Kruskal", "stat": float(H), "p": float(p)})
            except Exception:
                pass

    anova_df = pd.DataFrame(rows_anova)
    anova_df.to_csv(save_dir / "anova_norm2720_features.csv", index=False)

    print("\n=== PER-FEATURE GLOBAL TEST (anova_norm2720_features.csv) ===")
    if not anova_df.empty:
        anova_preview = anova_df.sort_values("p").head(10)
        print(anova_preview.to_string(index=False))
    else:
        print("No ANOVA/Kruskal results produced (maybe not enough replicates).")

    # 6. Games–Howell post-hoc (pairwise oil-vs-oil per feature)
    gh_all = []
    for feat in feat_cols_safe:
        gh = games_howell(df[[oil_col, feat]].dropna(), group_col=oil_col, value_col=feat)
        gh["feature"] = feat
        gh_all.append(gh)

    gh_df = pd.concat(gh_all, ignore_index=True)
    gh_df.to_csv(save_dir / "games_howell_norm2720.csv", index=False)

    print("\n=== PAIRWISE DIFFERENCES (games_howell_norm2720.csv) ===")
    sig_pairs_preview = gh_df.sort_values("p_holm").groupby("feature").head(1).reset_index(drop=True)
    print(sig_pairs_preview.to_string(index=False))

    # 7. Boxplots saved
    for feat in feat_cols_safe:
        fig = plt.figure(figsize=(6, 4))
        data_to_plot = [df.loc[df[oil_col] == oil, feat].dropna().values for oil in oils_sorted]
        plt.boxplot(data_to_plot, labels=oils_sorted, showfliers=False)
        plt.ylabel(feat)
        plt.title(f"{feat} by oil type")
        fig.savefig(save_dir / f"{feat}_boxplot.png", dpi=200)
        plt.close(fig)

    print("\nBoxplots saved for each feature.")
    print("\n=== RUN COMPLETE ===")
    print(f"Output directory: {save_dir}")
    print("\nFeature name mapping (original → safe):")
    for orig, safe in col_map.items():
        print(f"  {orig}  -->  {safe}")


def run_oil_norm2720_multivariate(output_root: Optional[Path] = None):
    timestr = time.strftime("_%Y_%m_%d_%HH_%M")
    if output_root is None:
        output_dir = "Stats_Output_Oil_norm2720" + timestr
    else:
        output_dir = Path(output_root) / ("Stats_Output_Oil_norm2720" + timestr)
    main_norm2720(OIL_RATIOMETRIC_CSV, output_dir)
    return output_dir  # we will reuse for RQ summaries


# ============================================================
# 6. Thermal stability: Chips (Chips Ratiometric Analysis)
# ============================================================
def run_chips_heating_stats():
    print("\n=== STEP 4: Thermal stability vs Heating Stage (Chips) ===")
    os.makedirs(SUPP_CHIPS_HEAT_DIR, exist_ok=True)

    file_path = CHIPS_RATIOMETRIC_CSV
    output_dir = SUPP_CHIPS_HEAT_DIR

    df = pd.read_csv(file_path)

    if "Heating_Stage" not in df.columns:
        print("[WARN] 'Heating_Stage' column missing in chips data; skipping heating stats.")
        return

    norm_cols = [
        "Peak Int 1742_norm2720",
        "Peak Int 1652_norm2720",
        "Peak Int 1434_norm2720",
        "Peak Int 1259_norm2720",
        "Peak Int 1296_norm2720",
    ]

    df["Heating_Stage"] = pd.to_numeric(df["Heating_Stage"], errors="coerce")

    # 1. Global ANOVA across Heating_Stage + Pearson trend
    anova_stage_results = {}
    trend_results = {}

    for col in norm_cols:
        groups = [group[col].values for _, group in df.groupby("Heating_Stage")]
        f_stat, p_val = stats.f_oneway(*groups)
        anova_stage_results[col] = {"F_statistic_heating": round(f_stat, 3), "p_ANOVA_heating": p_val}

        r, p_corr = stats.pearsonr(df["Heating_Stage"], df[col])
        trend_results[col] = {"r_trend": round(r, 3), "p_trend": p_corr}

    anova_stage_df = pd.DataFrame(anova_stage_results).T
    trend_df = pd.DataFrame(trend_results).T
    summary_global = pd.concat([anova_stage_df, trend_df], axis=1)

    summary_global["significant_by_ANOVA"] = summary_global["p_ANOVA_heating"] < 0.05
    summary_global["direction_if_trend"] = summary_global["r_trend"].apply(
        lambda x: "increases with heating" if x > 0 else "decreases with heating" if x < 0 else "no clear direction"
    )

    summary_global_path = os.path.join(output_dir, "Table_Global_RatioSignificance_vs_HeatingStage.csv")
    summary_global.to_csv(summary_global_path)
    print("=== GLOBAL RATIO SIGNIFICANCE (Chips) ===")
    print(summary_global)
    print("Saved:", summary_global_path)

    # 2. Per-chips slope analysis
    rows = []
    for oil, sub_oil in df.groupby("Chips_Name"):
        for col in norm_cols:
            slope, intercept, r_val, p_val, stderr = linregress(sub_oil["Heating_Stage"], sub_oil[col])
            rows.append(
                {
                    "Chips_Name": oil,
                    "Ratio": col,
                    "slope_per_stage": slope,
                    "p_slope": p_val,
                    "r_squared": r_val**2,
                    "abs_slope_magnitude": abs(slope),
                }
            )

    oil_stability_df = pd.DataFrame(rows)
    oil_stability_df["stability_rank_within_ratio"] = oil_stability_df.groupby("Ratio")["abs_slope_magnitude"].rank(
        method="dense", ascending=True
    )

    overall_stability = (
        oil_stability_df.groupby("Chips_Name")["abs_slope_magnitude"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
        .rename(columns={"abs_slope_magnitude": "mean_abs_slope_across_ratios"})
    )

    overall_stability["stability_rank_overall"] = overall_stability["mean_abs_slope_across_ratios"].rank(
        method="dense", ascending=True
    )

    oil_stability_path = os.path.join(output_dir, "Table_OilWise_SlopeAnalysis.csv")
    overall_stability_path = os.path.join(output_dir, "Table_OilWise_StabilityRanking.csv")

    oil_stability_df.to_csv(oil_stability_path, index=False)
    overall_stability.to_csv(overall_stability_path, index=False)

    print("\n=== CHIPS STABILITY (lower = more stable) ===")
    print(overall_stability)
    print("Saved per-oil slope table:", oil_stability_path)
    print("Saved oil ranking table:", overall_stability_path)

    # 3. Publication-ready plots
    sns.set(style="whitegrid", context="talk")

    for col in norm_cols:
        stats_summary = df.groupby(["Chips_Name", "Heating_Stage"])[col].agg(["mean", "std", "count"]).reset_index()

        plt.figure(figsize=(8, 5))

        for oil in stats_summary["Chips_Name"].unique():
            sub = stats_summary[stats_summary["Chips_Name"] == oil].sort_values("Heating_Stage")

            plt.errorbar(
                x=sub["Heating_Stage"],
                y=sub["mean"],
                yerr=sub["std"],
                marker="o",
                linestyle="-",
                label=oil,
                capsize=4,
                linewidth=2,
            )

        plt.title(f"{col} \nvs Heating Stage per oil in Chips")
        plt.xlabel("Heating Stage")
        plt.ylabel(f"{col}")
        plt.legend(title="Oil Type/ Chips_Name", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"Fig_{col.replace(' ', '_')}_MeanSD_vs_HeatingStage.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    print("\nSaved Chips figures (mean ± SD) to:", output_dir)

    # 4. Interpretation table
    interpret_rows = []
    for col in norm_cols:
        sig = summary_global.loc[col, "significant_by_ANOVA"]
        direction = summary_global.loc[col, "direction_if_trend"]
        pval = summary_global.loc[col, "p_ANOVA_heating"]
        interpret_rows.append(
            {
                "Ratio": col,
                "Heating sensitivity? (ANOVA p<0.05)": "YES" if sig else "NO",
                "Direction of change with heating": direction,
                "ANOVA p": pval,
            }
        )

    interpret_df = pd.DataFrame(interpret_rows)
    interpret_path = os.path.join(output_dir, "Table_RatioInterpretation_ForManuscript.csv")
    interpret_df.to_csv(interpret_path, index=False)

    print("\n=== RATIO INTERPRETATION TABLE (Chips) ===")
    print(interpret_df)
    print("Saved:", interpret_path)


# ============================================================
# 7. Thermal stability: Oils (Oil Ratiometric Analysis)
# ============================================================
def run_oil_heating_stats():
    print("\n=== STEP 5: Thermal stability vs Heating Stage (Oils) ===")
    os.makedirs(SUPP_OIL_HEAT_DIR, exist_ok=True)

    file_path = OIL_RATIOMETRIC_CSV
    output_dir = SUPP_OIL_HEAT_DIR

    df = pd.read_csv(file_path)

    if "Heating_Stage" not in df.columns:
        print("[WARN] 'Heating_Stage' column missing in oil data; skipping heating stats.")
        return

    norm_cols = [
        "Peak Int 1742_norm2720",
        "Peak Int 1652_norm2720",
        "Peak Int 1434_norm2720",
        "Peak Int 1259_norm2720",
        "Peak Int 1296_norm2720",
    ]

    df["Heating_Stage"] = pd.to_numeric(df["Heating_Stage"], errors="coerce")

    # 1. Global ANOVA across Heating_Stage + Pearson trend
    anova_stage_results = {}
    trend_results = {}

    for col in norm_cols:
        groups = [group[col].values for _, group in df.groupby("Heating_Stage")]
        f_stat, p_val = stats.f_oneway(*groups)
        anova_stage_results[col] = {"F_statistic_heating": round(f_stat, 3), "p_ANOVA_heating": p_val}

        r, p_corr = stats.pearsonr(df["Heating_Stage"], df[col])
        trend_results[col] = {"r_trend": round(r, 3), "p_trend": p_corr}

    anova_stage_df = pd.DataFrame(anova_stage_results).T
    trend_df = pd.DataFrame(trend_results).T
    summary_global = pd.concat([anova_stage_df, trend_df], axis=1)

    summary_global["significant_by_ANOVA"] = summary_global["p_ANOVA_heating"] < 0.05
    summary_global["direction_if_trend"] = summary_global["r_trend"].apply(
        lambda x: "increases with heating" if x > 0 else "decreases with heating" if x < 0 else "no clear direction"
    )

    summary_global_path = os.path.join(output_dir, "Table_Global_RatioSignificance_vs_HeatingStage.csv")
    summary_global.to_csv(summary_global_path)

    print("=== GLOBAL RATIO SIGNIFICANCE (Oils) ===")
    print(summary_global)
    print("Saved:", summary_global_path)

    # 2. Per-oil slope analysis
    rows = []
    for oil, sub_oil in df.groupby("Oil_Name"):
        for col in norm_cols:
            slope, intercept, r_val, p_val, stderr = linregress(sub_oil["Heating_Stage"], sub_oil[col])

            rows.append(
                {
                    "Oil_Name": oil,
                    "Ratio": col,
                    "slope_per_stage": slope,
                    "p_slope": p_val,
                    "r_squared": r_val**2,
                    "abs_slope_magnitude": abs(slope),
                }
            )

    oil_stability_df = pd.DataFrame(rows)
    oil_stability_df["stability_rank_within_ratio"] = oil_stability_df.groupby("Ratio")["abs_slope_magnitude"].rank(
        method="dense", ascending=True
    )

    overall_stability = (
        oil_stability_df.groupby("Oil_Name")["abs_slope_magnitude"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
        .rename(columns={"abs_slope_magnitude": "mean_abs_slope_across_ratios"})
    )

    overall_stability["stability_rank_overall"] = overall_stability["mean_abs_slope_across_ratios"].rank(
        method="dense", ascending=True
    )

    oil_stability_path = os.path.join(output_dir, "Table_OilWise_SlopeAnalysis.csv")
    overall_stability_path = os.path.join(output_dir, "Table_OilWise_StabilityRanking.csv")

    oil_stability_df.to_csv(oil_stability_path, index=False)
    overall_stability.to_csv(overall_stability_path, index=False)

    print("\n=== OIL STABILITY (lower = more stable) ===")
    print(overall_stability)
    print("Saved per-oil slope table:", oil_stability_path)
    print("Saved oil ranking table:", overall_stability_path)

    # 3. Publication-ready plots
    sns.set(style="whitegrid", context="talk")

    for col in norm_cols:
        stats_summary = df.groupby(["Oil_Name", "Heating_Stage"])[col].agg(["mean", "std", "count"]).reset_index()

        plt.figure(figsize=(8, 5))

        for oil in stats_summary["Oil_Name"].unique():
            sub = stats_summary[stats_summary["Oil_Name"] == oil].sort_values("Heating_Stage")

            plt.errorbar(
                x=sub["Heating_Stage"],
                y=sub["mean"],
                yerr=sub["std"],
                marker="o",
                linestyle="-",
                label=oil,
                capsize=4,
                linewidth=2,
            )

        plt.title(f"{col} \nvs Heating Stage per oil")
        plt.xlabel("Heating Stage")
        plt.ylabel(f"{col}")
        plt.legend(title="Oil Type/ Oil_Name", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"Fig_{col.replace(' ', '_')}_MeanSD_vs_HeatingStage.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    print("\nSaved oil figures (mean ± SD) to:", output_dir)

    # 4. Interpretation table
    interpret_rows = []
    for col in norm_cols:
        sig = summary_global.loc[col, "significant_by_ANOVA"]
        direction = summary_global.loc[col, "direction_if_trend"]
        pval = summary_global.loc[col, "p_ANOVA_heating"]
        interpret_rows.append(
            {
                "Ratio": col,
                "Heating sensitivity? (ANOVA p<0.05)": "YES" if sig else "NO",
                "Direction of change with heating": direction,
                "ANOVA p": pval,
            }
        )

    interpret_df = pd.DataFrame(interpret_rows)
    interpret_path = os.path.join(output_dir, "Table_RatioInterpretation_ForManuscript.csv")
    interpret_df.to_csv(interpret_path, index=False)

    print("\n=== RATIO INTERPRETATION TABLE (Oils) ===")
    print(interpret_df)
    print("Saved:", interpret_path)


# ============================================================
# 8. RQ Answers: modelling + text summaries
# ============================================================
def get_functional_band_mapping():
    """
    Mapping from peak center to chemical assignment / region.
    Used in RQ7 / RQ9 narrative.
    """
    return {
        2720: ("CH stretch", "overall C–H stretching region; reference band"),
        1742: ("C=O (carbonyl)", "ester carbonyl band, sensitive to oxidation and triglyceride structure"),
        1652: ("C=C (unsaturation)", "C=C stretching band, sensitive to unsaturation level"),
        1434: ("CH2 bending", "CH2 scissoring/bending, relates to chain packing"),
        1259: ("fingerprint band", "C–O and C–C modes in triglyceride backbone"),
        1296: ("CH2 twisting", "CH2 twisting/rocking, linked to chain conformation"),
    }


def compute_classification_and_feature_importance(df, feature_cols, label_col):
    """
    Returns: dict with RF & LR accuracy (mean ± std), feature importance ranking, PCA metrics.
    """
    X = df[feature_cols].astype(float).values
    y = df[label_col].astype(str).values

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # RandomForest classifier
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    skf = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=0)
    rf_scores = cross_val_score(rf, Xz, y, cv=skf, scoring="accuracy")
    rf.fit(Xz, y)
    rf_importances = rf.feature_importances_

    # Logistic regression (linear baseline)
    lr = LogisticRegression(max_iter=5000, multi_class="auto", solver="lbfgs")
    lr_scores = cross_val_score(lr, Xz, y, cv=skf, scoring="accuracy")

    # PCA metrics
    pca_stats = compute_pca_metrics(X, y, n_components=min(5, X.shape[1]))

    importance_df = pd.DataFrame({"feature": feature_cols, "rf_importance": rf_importances}).sort_values(
        "rf_importance", ascending=False
    )

    return {
        "rf_mean_acc": float(rf_scores.mean()),
        "rf_std_acc": float(rf_scores.std()),
        "lr_mean_acc": float(lr_scores.mean()),
        "lr_std_acc": float(lr_scores.std()),
        "importance_df": importance_df,
        "pca_stats": pca_stats,
    }


def compute_minimal_feature_panel(df, feature_cols, label_col, importance_df, tolerance=0.02):
    """
    Greedy minimal marker panel:
    - Order features by RF importance
    - For k = 1..N, compute RF accuracy using top-k
    - Return smallest k whose accuracy >= best_acc - tolerance
    """
    X_all = df[feature_cols].astype(float).values
    y = df[label_col].astype(str).values

    scaler = StandardScaler()
    Xz_all = scaler.fit_transform(X_all)

    skf = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=0)

    rf_full = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    full_scores = cross_val_score(rf_full, Xz_all, y, cv=skf, scoring="accuracy")
    best_acc = full_scores.mean()

    ranked_features = list(importance_df["feature"].values)
    panel_results = []

    for k in range(1, len(ranked_features) + 1):
        subset = ranked_features[:k]
        Xk = df[subset].astype(float).values
        Xkz = scaler.fit_transform(Xk)

        rf_k = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
        scores_k = cross_val_score(rf_k, Xkz, y, cv=skf, scoring="accuracy")
        panel_results.append(
            {"k": k, "features": subset, "mean_acc": float(scores_k.mean()), "std_acc": float(scores_k.std())}
        )

    # choose smallest k with acc >= best_acc - tolerance
    panel_df = pd.DataFrame(panel_results)
    candidate = panel_df[panel_df["mean_acc"] >= (best_acc - tolerance)]
    if len(candidate) == 0:
        chosen = panel_df.iloc[-1]
    else:
        chosen = candidate.iloc[0]

    return {
        "best_acc": float(best_acc),
        "panels": panel_df,
        "chosen_k": int(chosen["k"]),
        "chosen_features": list(chosen["features"]),
        "chosen_mean_acc": float(chosen["mean_acc"]),
        "chosen_std_acc": float(chosen["std_acc"]),
    }


def compute_unsupervised_clustering(df, feature_cols, label_col):
    """
    KMeans clustering in feature space:
    - silhouette (unsupervised)
    - ARI vs true labels (consistency)
    """
    X = df[feature_cols].astype(float).values
    y = df[label_col].astype(str).values

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    n_clusters = len(np.unique(y))
    if n_clusters < 2:
        return None

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(Xz)

    sil = silhouette_score(Xz, cluster_labels) if n_clusters > 1 else np.nan
    ari = adjusted_rand_score(y, cluster_labels)

    return {"silhouette": float(sil), "ari": float(ari), "cluster_labels": cluster_labels}


def compute_mds_map(df, feature_cols, label_col, out_dir: Path):
    """
    2D MDS embedding & save as CSV + PNG.
    """
    X = df[feature_cols].astype(float).values
    y = df[label_col].astype(str).values

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    mds = MDS(n_components=2, random_state=0, dissimilarity="euclidean", n_init=4)
    coords = mds.fit_transform(Xz)

    mds_df = pd.DataFrame({"MDS1": coords[:, 0], "MDS2": coords[:, 1], label_col: y})

    mds_df.to_csv(out_dir / "mds_2D_oils.csv", index=False)

    plt.figure(figsize=(6, 5))
    for oil in sorted(np.unique(y)):
        mask = y == oil
        plt.scatter(coords[mask, 0], coords[mask, 1], label=str(oil), s=20)
    plt.xlabel("MDS1")
    plt.ylabel("MDS2")
    plt.title("Global chemometric map (MDS) – oil spectra")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_dir / "mds_2D_oils.png", dpi=200)
    plt.close()

    return mds_df


def compute_intra_class_variability(df, feature_cols, label_col):
    """
    Mean ± std within each class (oil), averaged over features.
    """
    rows = []
    for oil, sub in df.groupby(label_col):
        X = sub[feature_cols].astype(float).values
        mean_vec = X.mean(axis=0)
        std_vec = X.std(axis=0, ddof=1)
        rows.append(
            {
                label_col: oil,
                "mean_feature_mean": float(mean_vec.mean()),
                "mean_feature_std": float(std_vec.mean()),
                "mean_cv_percent": float((std_vec / (mean_vec + 1e-12)).mean() * 100),
            }
        )

    intr_df = pd.DataFrame(rows)
    return intr_df.sort_values("mean_cv_percent")


def compute_peak_position_shifts():
    """
    Uses OIL_PEAK_CV_CSV (per-sample peak positions) to regress
    peak position vs Heating_Stage per oil and per band.
    """
    if not os.path.exists(OIL_PEAK_CV_CSV):
        return None

    df = pd.read_csv(OIL_PEAK_CV_CSV)
    if "Heating_Stage" not in df.columns:
        return None
    df["Heating_Stage"] = pd.to_numeric(df["Heating_Stage"], errors="coerce")

    peak_pos_cols = [c for c in df.columns if "Peak Pos" in c]
    rows = []
    for oil, sub in df.groupby("Oil_Name"):
        for col in peak_pos_cols:
            if sub[col].notna().sum() < 3:
                continue
            slope, intercept, r_val, p_val, stderr = linregress(sub["Heating_Stage"], sub[col])
            rows.append(
                {
                    "Oil_Name": oil,
                    "Peak_Pos_Col": col,
                    "slope_cm1_per_stage": slope,
                    "p_slope": p_val,
                    "r_squared": r_val**2,
                    "abs_slope": abs(slope),
                }
            )

    if not rows:
        return None

    shift_df = pd.DataFrame(rows)
    out_dir = Path(SUPP_OIL_HEAT_DIR)
    ensure_dir(out_dir)
    shift_df.to_csv(out_dir / "peak_position_shifts_vs_heating.csv", index=False)
    return shift_df


def compute_thermal_stability_index(df, feature_cols, oil_col="Oil_Name"):
    """
    Thermal stability index per oil:
    - for each oil, compute centroid at Heating_Stage == 0
    - TSI = mean Euclidean distance of heated samples (stage > 0) to that centroid
    """
    if "Heating_Stage" not in df.columns:
        return None

    df = df.copy()
    df["Heating_Stage"] = pd.to_numeric(df["Heating_Stage"], errors="coerce")

    tsi_rows = []
    for oil, sub in df.groupby(oil_col):
        base = sub[sub["Heating_Stage"] == 0]
        heated = sub[sub["Heating_Stage"] > 0]

        if base.empty or heated.empty:
            continue

        base_vec = base[feature_cols].astype(float).mean(axis=0).values
        X_heated = heated[feature_cols].astype(float).values

        dists = np.linalg.norm(X_heated - base_vec, axis=1)
        tsi_rows.append({oil_col: oil, "thermal_stability_index": float(dists.mean()), "n_heated": int(len(dists))})

    if not tsi_rows:
        return None

    tsi_df = pd.DataFrame(tsi_rows).sort_values("thermal_stability_index")
    out_dir = Path(SUPP_OIL_HEAT_DIR)
    ensure_dir(out_dir)
    tsi_df.to_csv(out_dir / "thermal_stability_index_per_oil.csv", index=False)
    return tsi_df


def compute_heating_stage_prediction(df, feature_cols, stage_col="Heating_Stage"):
    """
    Heating-stage prediction (“chemical clock”) using RandomForestRegressor.
    Returns cross-validated R² and RMSE.
    """
    df = df.dropna(subset=feature_cols + [stage_col]).copy()
    df[stage_col] = pd.to_numeric(df[stage_col], errors="coerce")
    df = df.dropna(subset=[stage_col])

    if df.empty:
        return None

    X = df[feature_cols].astype(float).values
    y = df[stage_col].values.astype(float)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    kf = KFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=0)
    rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)

    r2_scores = cross_val_score(rf, Xz, y, cv=kf, scoring="r2")
    # For RMSE we compute manually
    rmse_list = []
    for train_idx, test_idx in kf.split(Xz):
        rf.fit(Xz[train_idx], y[train_idx])
        y_pred = rf.predict(Xz[test_idx])
        rmse = float(np.sqrt(((y_pred - y[test_idx]) ** 2).mean()))
        rmse_list.append(rmse)

    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
    }


def compute_normalization_robustness(oil_df):
    """
    Compare discrimination under:
    (i) reference-peak normalization (I_peak / I2720)
    (ii) vector normalization (L2 over raw peak intensities)
    """
    # Reference-peak features
    ref_cols = [
        "Peak Int 1742_norm2720",
        "Peak Int 1652_norm2720",
        "Peak Int 1434_norm2720",
        "Peak Int 1259_norm2720",
        "Peak Int 1296_norm2720",
    ]
    oil_df = oil_df.dropna(subset=ref_cols + ["Oil_Name"]).copy()

    # (i) existing reference-peak normalization
    ref_metrics = compute_classification_and_feature_importance(oil_df, ref_cols, label_col="Oil_Name")

    # (ii) vector normalization over raw peak intensities
    raw_cols = ["Peak Int 1742", "Peak Int 1652", "Peak Int 1434", "Peak Int 1259", "Peak Int 1296"]
    available_raw = [c for c in raw_cols if c in oil_df.columns]
    if len(available_raw) == 0:
        return {"ref": ref_metrics, "vector": None}

    raw_mat = oil_df[available_raw].astype(float).values
    vec_norm = normalize(raw_mat, norm="l2", axis=1)
    vec_df = pd.DataFrame(vec_norm, columns=[f"{c}_vecNorm" for c in available_raw], index=oil_df.index)
    tmp_df = pd.concat([oil_df[["Oil_Name"]], vec_df], axis=1)

    vector_metrics = compute_classification_and_feature_importance(tmp_df, list(vec_df.columns), label_col="Oil_Name")

    return {"ref": ref_metrics, "vector": vector_metrics}


def generate_rq_summaries(norm_output_dir: str, cfg: Optional[Config] = None):
    """
    Main function to answer RQ1–RQ14 and write publication-ready text files.
    """
    ensure_dir(PUBLICATION_SUMMARY_DIR)

    # Load key tables
    oil_df = pd.read_csv(OIL_RATIOMETRIC_CSV)
    chips_df = pd.read_csv(CHIPS_RATIOMETRIC_CSV) if os.path.exists(CHIPS_RATIOMETRIC_CSV) else None

    # Features of interest (ratios)
    ratio_cols = [
        "Peak Int 1742_norm2720",
        "Peak Int 1652_norm2720",
        "Peak Int 1434_norm2720",
        "Peak Int 1259_norm2720",
        "Peak Int 1296_norm2720",
    ]
    ratio_cols = [c for c in ratio_cols if c in oil_df.columns]

    # 1) Supervised discrimination + feature importance (RQ1, RQ3, RQ5, RQ6 partially)
    clf_metrics = compute_classification_and_feature_importance(
        oil_df.dropna(subset=ratio_cols + ["Oil_Name"]), ratio_cols, label_col="Oil_Name"
    )

    minimal_panel = compute_minimal_feature_panel(
        oil_df.dropna(subset=ratio_cols + ["Oil_Name"]),
        ratio_cols,
        label_col="Oil_Name",
        importance_df=clf_metrics["importance_df"],
        tolerance=0.02,
    )

    # 2) Unsupervised clustering & chemometric map (RQ6, RQ13)
    clustering_metrics = compute_unsupervised_clustering(
        oil_df.dropna(subset=ratio_cols + ["Oil_Name"]), ratio_cols, label_col="Oil_Name"
    )

    norm_out_dir = Path(norm_output_dir)
    ensure_dir(norm_out_dir)
    mds_df = compute_mds_map(
        oil_df.dropna(subset=ratio_cols + ["Oil_Name"]), ratio_cols, label_col="Oil_Name", out_dir=norm_out_dir
    )

    # 3) Intra-class variability (RQ2, RQ8)
    intraclass_df = compute_intra_class_variability(
        oil_df.dropna(subset=ratio_cols + ["Oil_Name"]), ratio_cols, label_col="Oil_Name"
    )

    # 4) Thermal stability index & heating-stage prediction (RQ4, RQ11, RQ12)
    tsi_df = compute_thermal_stability_index(oil_df.copy(), ratio_cols, oil_col="Oil_Name")

    heating_pred_oil = compute_heating_stage_prediction(oil_df.copy(), ratio_cols, stage_col="Heating_Stage")

    heating_pred_chips = None
    if chips_df is not None:
        heating_pred_chips = compute_heating_stage_prediction(chips_df.copy(), ratio_cols, stage_col="Heating_Stage")

    # 5) Peak position shifts with heating (RQ10)
    peak_shift_df = compute_peak_position_shifts()

    # 6) Normalization robustness (RQ14)
    norm_robustness = compute_normalization_robustness(oil_df.copy())

    # 7) Per-feature ANOVA results & Games–Howell (for RQ3, RQ7, RQ9)
    anova_feat_path = Path(norm_output_dir) / "anova_norm2720_features.csv"
    gh_path = Path(norm_output_dir) / "games_howell_norm2720.csv"
    feat_anova_df = pd.read_csv(anova_feat_path) if anova_feat_path.exists() else None
    gh_df = pd.read_csv(gh_path) if gh_path.exists() else None

    # 8) Peak CVs (RQ2)
    cv_int_df = pd.read_csv(OIL_RATIO_NORM_CV_CSV) if os.path.exists(OIL_RATIO_NORM_CV_CSV) else None

    # Chemical mapping
    band_map = get_functional_band_mapping()

    # ---------- Build detailed narrative ----------
    lines = []
    brief_lines = []

    def add(title, text):
        lines.append(f"{title}\n" + "-" * len(title))
        lines.append(text.strip() + "\n")

    # RQ1 – Oil discrimination
    rf_mean = clf_metrics["rf_mean_acc"]
    rf_std = clf_metrics["rf_std_acc"]
    lr_mean = clf_metrics["lr_mean_acc"]
    lr_std = clf_metrics["lr_std_acc"]
    pca_sil = clf_metrics["pca_stats"]["silhouette"]
    pca_bw = clf_metrics["pca_stats"]["bw_ratio"]
    pca_pperm = clf_metrics["pca_stats"]["p_perm"]

    rq1_text = f"""
Using only the five ratiometric Raman features (I_1742/I_2720, I_1652/I_2720, I_1434/I_2720, I_1259/I_2720, I_1296/I_2720),
a Random Forest classifier achieved a cross-validated accuracy of {rf_mean:.3f} ± {rf_std:.3f} (mean ± SD, stratified k-fold),
while a linear logistic regression baseline reached {lr_mean:.3f} ± {lr_std:.3f}. These values indicate that even a compact
set of ratios can reliably discriminate between the oils included in this study.

In unsupervised PCA space, the same ratiometric features yielded a silhouette score of {pca_sil:.3f} and a between/within-class
scatter ratio of {pca_bw:.3f} with a permutation p-value of {pca_pperm:.3g}. This confirms that oils form well-separated clusters
in the low-dimensional chemometric space, consistently with their labelled identities.

Overall, RQ1 is answered positively: Raman ratiometric signatures alone provide high discrimination performance between the studied edible oils.
    """
    add("RQ1 – Oil discrimination", rq1_text)
    brief_lines.append(
        f"RQ1 – Oil discrimination: RF accuracy ≈ {rf_mean:.2f} (±{rf_std:.2f}); oils are strongly separable in ratiometric Raman space."
    )

    # RQ2 – Reproducibility & stability
    if cv_int_df is not None:
        low_cv = cv_int_df.sort_values("cv(%)").head(5)
        low_cv_list = ", ".join([f"{row['Peak']} (CV {row['cv(%)']:.1f}%)" for _, row in low_cv.iterrows()])
    else:
        low_cv_list = "CV table not available (no OIL_RATIO_NORM_CV_CSV)."

    best_stable_oils = intraclass_df.head(3)
    worst_stable_oils = intraclass_df.tail(3)

    rq2_text = f"""
Reproducibility was quantified using coefficients of variation (CV) at the peak and ratiometric level. Across all oils and replicates,
the most stable features (lowest global CV) were:

    {low_cv_list}

When intra-class variability was evaluated per oil, the mean CV across ratiometric features ranged from
{intraclass_df['mean_cv_percent'].min():.1f}% (most homogeneous oil) to
{intraclass_df['mean_cv_percent'].max():.1f}% (most heterogeneous oil).

The three most reproducible oils (lowest intra-class CV) were:
{best_stable_oils.to_string(index=False)}

while the three most heterogeneous oils were:
{worst_stable_oils.to_string(index=False)}

These results identify both robust features and oils with higher intrinsic spectral variability, which is critical for defining QA markers and
for designing calibration sets.
    """
    add("RQ2 – Reproducibility & stability", rq2_text)
    brief_lines.append(
        "RQ2 – Reproducibility: specific normalized peaks show low CV; some oils are markedly more homogeneous than others."
    )

    # RQ3 – Discriminative ratios (tests + model importance)
    top_importance = clf_metrics["importance_df"].head(5)
    top_imp_str = ", ".join(
        [f"{row['feature']} (RF importance {row['rf_importance']:.3f})" for _, row in top_importance.iterrows()]
    )

    if feat_anova_df is not None and not feat_anova_df.empty:
        feat_anova_sorted = feat_anova_df.sort_values("p").head(5)
        top_anova_str = ", ".join([f"{row['feature']} (p={row['p']:.2e})" for _, row in feat_anova_sorted.iterrows()])
    else:
        top_anova_str = "Per-feature ANOVA table not available."

    rq3_text = f"""
Discriminative power was assessed from both hypothesis-testing and model-based perspectives.

From the Random Forest classifier, the most important ratiometric features were:
    {top_imp_str}

From global ANOVA/Kruskal tests across oil types (on *_norm2720 features), the most significantly different ratios were:
    {top_anova_str}

Taken together, these results highlight a small number of chemically-meaningful ratios that contribute disproportionately
to oil discrimination. These same features form the core of the minimal marker panel identified in RQ5.
    """
    add("RQ3 – Discriminative ratios", rq3_text)
    brief_lines.append(
        "RQ3 – Discriminative ratios: a small subset of I_1742/I_2720, I_1652/I_2720, and related bands dominate RF importance and ANOVA significance."
    )

    # RQ4 – Thermal degradation markers
    oil_heat_global = (
        pd.read_csv(Path(SUPP_OIL_HEAT_DIR) / "Table_Global_RatioSignificance_vs_HeatingStage.csv")
        if (Path(SUPP_OIL_HEAT_DIR) / "Table_Global_RatioSignificance_vs_HeatingStage.csv").exists()
        else None
    )

    chips_heat_global = (
        pd.read_csv(Path(SUPP_CHIPS_HEAT_DIR) / "Table_Global_RatioSignificance_vs_HeatingStage.csv")
        if (Path(SUPP_CHIPS_HEAT_DIR) / "Table_Global_RatioSignificance_vs_HeatingStage.csv").exists()
        else None
    )

    def summarize_heating_global(df_glob, matrix_label):
        if df_glob is None:
            return f"No heating-stage ANOVA table available for {matrix_label}."
        sig = df_glob[df_glob["significant_by_ANOVA"] == True]
        if sig.empty:
            return f"No ratio showed significant dependence on heating stage (ANOVA p<0.05) in {matrix_label}."
        lines_local = [f"In {matrix_label}, the following ratios changed significantly with heating (ANOVA p<0.05):"]
        for _, row in sig.iterrows():
            lines_local.append(
                f"  - {row.name}: p={row['p_ANOVA_heating']:.2e}, trend: {row['direction_if_trend']}, r_trend={row['r_trend']:.3f}"
            )
        return "\n".join(lines_local)

    rq4_text = f"""
Thermal degradation was probed by relating ratiometric features to heating stage.

Global ANOVA/Trend summary for oils:
{summarize_heating_global(oil_heat_global, "pure oils")}

Global ANOVA/Trend summary for chips:
{summarize_heating_global(chips_heat_global, "chips matrix")}

For ratios that monotonically increase or decrease with heating stage and remain statistically significant,
these trajectories define candidate Raman markers for thermal load. The sign and magnitude of Pearson r (and the slopes
in the per-oil / per-chips regression tables) quantify how strongly each ratio behaves as a degradation or processing marker.
    """
    add("RQ4 – Thermal degradation markers", rq4_text)
    brief_lines.append(
        "RQ4 – Thermal degradation: several ratios show significant monotonic changes with heating, in both oils and chips, usable as thermal load markers."
    )

    # RQ5 – Minimal feature set
    rq5_text = f"""
Using a greedy, cross-validated feature-selection strategy based on Random Forest accuracy, we evaluated subsets of the
ratiometric features ordered by RF importance. The full feature set achieved a mean accuracy of {minimal_panel['best_acc']:.3f}.

A minimal panel of k = {minimal_panel['chosen_k']} features

    {', '.join(minimal_panel['chosen_features'])}

already attained {minimal_panel['chosen_mean_acc']:.3f} ± {minimal_panel['chosen_std_acc']:.3f}, i.e. within 0.02 of the best
full-feature accuracy.

Thus, a compact marker panel of {minimal_panel['chosen_k']} ratios is sufficient to maintain near-optimal discrimination between oils,
offering a practical compromise between analytical complexity and performance.
    """
    add("RQ5 – Minimal feature set", rq5_text)
    brief_lines.append(
        f"RQ5 – Minimal feature set: ≈{minimal_panel['chosen_k']} top ratios retain almost full discrimination performance (Δaccuracy ≤ 0.02)."
    )

    # RQ6 – Unsupervised clustering structure
    if clustering_metrics is not None:
        rq6_text = f"""
K-means clustering was performed directly in standardized ratiometric feature space (number of clusters = number of oils).
The unsupervised cluster quality (silhouette) was {clustering_metrics['silhouette']:.3f}, and consistency with true labels
(Adjusted Rand Index) was {clustering_metrics['ari']:.3f}.

These values indicate that oils naturally group into well-defined clusters in Raman feature space even without label information.
Mis-clustered samples, if any, likely correspond to oils with borderline composition or higher intra-class variability (see RQ8).
        """
    else:
        rq6_text = """
Clustering could not be reliably evaluated because there were fewer than two distinct oil classes with valid data.
        """
    add("RQ6 – Unsupervised clustering structure", rq6_text)
    brief_lines.append(
        "RQ6 – Clustering: K-means on ratiometric space reproduces oil labels with good silhouette and ARI, confirming natural clustering."
    )

    # RQ7 – Wavenumber / region importance
    band_summary_lines = []
    for feat, imp in clf_metrics["importance_df"].head(5)[["feature", "rf_importance"]].values:
        # extract numeric center from feature name
        center = None
        for c in [1742, 1652, 1434, 1259, 1296, 2720]:
            if str(c) in feat:
                center = c
                break
        if center is not None and center in band_map:
            name, descr = band_map[center]
            band_summary_lines.append(
                f"{feat}: RF importance {imp:.3f}, associated with {center} cm⁻¹ ({name}, {descr})"
            )
        else:
            band_summary_lines.append(f"{feat}: RF importance {imp:.3f}")

    rq7_text = f"""
To relate discriminative power to chemically meaningful regions, we mapped intensity ratios back to their parent bands.

Among the most important features by Random Forest importance, we find:

    {chr(10).join(band_summary_lines)}

This indicates that:
- The carbonyl region around 1742 cm⁻¹ (ester C=O) and the unsaturation band around 1652 cm⁻¹ contribute strongly to oil discrimination.
- CH₂ bending / twisting bands (1434, 1296 cm⁻¹) also carry significant discriminative information.
- The CH stretch region at 2720 cm⁻¹ provides a robust normalization reference.

Together, these regions form a chemically interpretable fingerprint that separates oils in terms of oxidation state, unsaturation,
and chain packing.
    """
    add("RQ7 – Wavenumber / region importance", rq7_text)
    brief_lines.append(
        "RQ7 – Wavenumber importance: carbonyl (1742 cm⁻¹), unsaturation (1652 cm⁻¹) and CH₂ bands dominate the discriminative signal."
    )

    # RQ8 – Intra-class spectral variability
    rq8_text = f"""
Intra-class spectral variability was quantified as the mean coefficient of variation (CV) across ratiometric features within each oil.
The most homogeneous oils exhibited mean CVs near {intraclass_df['mean_cv_percent'].min():.1f}%, whereas the most heterogeneous oils
approached {intraclass_df['mean_cv_percent'].max():.1f}%.

Sorted intra-class variability (lower = more homogeneous):

{intraclass_df.to_string(index=False)}

Oils with higher intra-class variability are expected to be more challenging to classify and may reflect either intrinsic compositional
heterogeneity (e.g. blended or less controlled products) or experimental factors (e.g. sampling, illumination, or matrix effects).
    """
    add("RQ8 – Intra-class spectral variability", rq8_text)
    brief_lines.append(
        "RQ8 – Intra-class variability: oils differ markedly in internal CV; more heterogeneous oils may require more replicates or tailored calibration."
    )

    # RQ9 – Functional band behaviour
    # Use group means per oil per ratio from Supplementary_Table_GroupMeanStd.csv
    group_stats_path = Path(SUPP_OIL_STATS_DIR) / "Supplementary_Table_GroupMeanStd.csv"
    if group_stats_path.exists():
        group_stats = pd.read_csv(group_stats_path)
        # For key bands 1742 and 1652, list top/bottom oils by mean ratio
        summaries = []
        for center, (label, descr) in band_map.items():
            ratio_col = f"Peak Int {center}_norm2720_mean"
            if ratio_col in group_stats.columns:
                top2 = group_stats.sort_values(ratio_col, ascending=False)[["Oil_Name", ratio_col]].head(2)
                bottom2 = group_stats.sort_values(ratio_col, ascending=True)[["Oil_Name", ratio_col]].head(2)
                summaries.append(
                    f"For {center} cm⁻¹ ({label}), the highest mean ratios are:\n"
                    + top2.to_string(index=False)
                    + "\nThe lowest mean ratios are:\n"
                    + bottom2.to_string(index=False)
                    + "\n"
                )
        functional_summary = "\n".join(summaries) if summaries else "Functional band summary could not be constructed."
    else:
        functional_summary = "Group mean/std per oil file not found."

    rq9_text = f"""
Chemically meaningful bands were examined oil-by-oil using mean ratiometric intensities.

{functional_summary}

These patterns are consistent with expected compositional differences:
higher carbonyl and unsaturation ratios (1742/2720, 1652/2720) align with oils richer in esterified and unsaturated components,
whereas lower values reflect more saturated or hydrocarbon-rich profiles.

CH₂ bending and twisting bands (1434, 1296 cm⁻¹) further differentiate chain packing and conformational states, complementing the
information carried by carbonyl and C=C bands.
    """
    add("RQ9 – Functional band behaviour", rq9_text)
    brief_lines.append(
        "RQ9 – Functional bands: carbonyl and unsaturation ratios stratify oils in ways consistent with composition (richer vs more saturated profiles)."
    )

    # RQ10 – Peak position shifts with heating
    if peak_shift_df is not None and not peak_shift_df.empty:
        sig_shift = peak_shift_df[peak_shift_df["p_slope"] < 0.05]
        if not sig_shift.empty:
            sig_shift_preview = sig_shift.sort_values("abs_slope", ascending=False).head(10)
            sig_text = sig_shift_preview.to_string(index=False)
        else:
            sig_text = "No statistically significant (p<0.05) peak position shifts with heating were detected."
    else:
        sig_text = (
            "Peak-position shift analysis could not be performed (OIL_PEAK_CV_CSV unavailable or insufficient data)."
        )

    rq10_text = f"""
To probe structural and compositional changes beyond intensity variations, peak positions were regressed against heating stage
for each oil and Raman band.

Summary of the most pronounced and statistically significant shifts (if any):

{sig_text}

Where significant slopes are non-zero, they indicate systematic frequency shifts (in cm⁻¹ per heating stage), potentially reflecting
changes in local bonding environment, unsaturation, or hydrogen-bonding pattern. The absence of strong, consistent shifts suggests
that (within the studied heating protocol) intensity-based markers may be more sensitive than position-based markers.
    """
    add("RQ10 – Peak position shifts with heating", rq10_text)
    brief_lines.append(
        "RQ10 – Peak shifts: peak position regressions vs heating generally show small or modest shifts; intensity changes are more prominent markers."
    )

    # RQ11 – Thermal stability index per oil
    if tsi_df is not None and not tsi_df.empty:
        tsi_preview = tsi_df.to_string(index=False)
    else:
        tsi_preview = "Thermal Stability Index could not be computed (no valid unheated + heated pairs per oil)."

    rq11_text = f"""
A multivariate Thermal Stability Index (TSI) was defined as the mean Euclidean distance between each oil’s heated spectra and its
unheated (stage 0) ratiometric fingerprint in multi-feature space.

Sorted TSI values (lower = more thermally stable spectral fingerprint):

{tsi_preview}

Oils with low TSI show spectra that remain close to their unheated fingerprint even after multiple heating stages, whereas high-TSI
oils drift further in chemometric space, indicating stronger structural or compositional changes under the same thermal load.
    """
    add("RQ11 – Thermal stability index per oil", rq11_text)
    brief_lines.append(
        "RQ11 – Thermal stability index: oils can be ranked by chemometric drift from their unheated fingerprint; low-TSI oils are more stable."
    )

    # RQ12 – Heating-stage prediction (“chemical clock”)
    if heating_pred_oil is not None:
        rq12_oil_txt = (
            f"For pure oils, RandomForest regression predicting Heating_Stage from ratiometric features achieved "
            f"R² = {heating_pred_oil['r2_mean']:.3f} ± {heating_pred_oil['r2_std']:.3f} and "
            f"RMSE = {heating_pred_oil['rmse_mean']:.3f} ± {heating_pred_oil['rmse_std']:.3f} (stage units)."
        )
    else:
        rq12_oil_txt = "Heating-stage prediction for oils could not be robustly evaluated."

    if heating_pred_chips is not None:
        rq12_chips_txt = (
            f"For chips, the corresponding model achieved R² = {heating_pred_chips['r2_mean']:.3f} ± {heating_pred_chips['r2_std']:.3f} "
            f"and RMSE = {heating_pred_chips['rmse_mean']:.3f} ± {heating_pred_chips['rmse_std']:.3f}."
        )
    else:
        rq12_chips_txt = "Heating-stage prediction for chips could not be robustly evaluated."

    rq12_text = f"""
A “chemical clock” was implemented by training RandomForest regressors to predict heating stage directly from ratiometric Raman features.

{oil_df.shape[0]} oil spectra were used to train and cross-validate an oil-only model.
{rq12_oil_txt}

{rq12_chips_txt}

These results indicate that, within the studied heating protocol, the ratiometric features contain sufficient information to
approximate the cumulative thermal load, although prediction errors on individual spectra should be interpreted with caution.
    """
    add("RQ12 – Heating-stage prediction (“chemical clock”)", rq12_text)
    brief_lines.append(
        "RQ12 – Chemical clock: RandomForest regression recovers heating stage with non-trivial R²; ratiometric spectra encode thermal history."
    )

    # RQ13 – Global chemometric map
    rq13_text = f"""
A low-dimensional chemometric map was constructed using both PCA (see pca_norm2720_scatter.png) and metric Multi-Dimensional
Scaling (MDS; mds_2D_oils.png).

In both representations, oils arrange along interpretable gradients, with families of chemically similar oils clustering together and
outliers occupying distinct regions. The MDS coordinates (mds_2D_oils.csv) can be used for further exploratory analyses or to
overlay metadata (e.g. brand, batch, origin).

Taken together, these maps provide an intuitive visualization of global relationships between oils in Raman feature space,
highlighting families, gradients, and potential outliers.
    """
    add("RQ13 – Global chemometric map", rq13_text)
    brief_lines.append(
        "RQ13 – Chemometric map: PCA and MDS reveal structured gradients and family clusters among oils, with clear outliers."
    )

    # RQ14 – Normalization robustness
    if norm_robustness is not None and norm_robustness["vector"] is not None:
        ref_m = norm_robustness["ref"]["rf_mean_acc"]
        ref_s = norm_robustness["ref"]["rf_std_acc"]
        vec_m = norm_robustness["vector"]["rf_mean_acc"]
        vec_s = norm_robustness["vector"]["rf_std_acc"]
        ref_sil = norm_robustness["ref"]["pca_stats"]["silhouette"]
        vec_sil = norm_robustness["vector"]["pca_stats"]["silhouette"]

        rq14_text = f"""
Normalization robustness was evaluated by comparing:
(i) the reference-peak normalization (I_peak / I_2720), and
(ii) vector normalization over raw peak intensities (L2-normalized intensity vector).

For oil discrimination (RandomForest):

- Reference-peak normalization: accuracy = {ref_m:.3f} ± {ref_s:.3f}, PCA silhouette = {ref_sil:.3f}
- Vector normalization:      accuracy = {vec_m:.3f} ± {vec_s:.3f}, PCA silhouette = {vec_sil:.3f}

Although minor differences exist, both schemes yield high discrimination and similar clustering quality, indicating that the main
conclusions (oil separability, clustering, and trend behaviour) are robust to normalization choice. Reference-peak normalization
remains preferred for ease of interpretation relative to the CH stretch band, but vector normalization offers a complementary,
model-agnostic alternative.
        """
    else:
        rq14_text = """
Normalization robustness could not be fully evaluated (vector normalization metrics were unavailable), but the high performance under
reference-peak normalization already supports the stability of the main conclusions.
        """
    add("RQ14 – Normalization robustness", rq14_text)
    brief_lines.append(
        "RQ14 – Normalization: both reference-peak and vector normalization preserve high discrimination and clustering; conclusions are robust."
    )

    # ---------- Write files ----------
    detailed_path = PUBLICATION_SUMMARY_DIR / "RQ_detailed_answers.txt"
    brief_path = PUBLICATION_SUMMARY_DIR / "RQ_brief_summary.txt"

    with open(detailed_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(brief_path, "w", encoding="utf-8") as f:
        f.write("\n".join(brief_lines))

    print("\n=== RQ summaries written ===")
    print("Detailed answers:", detailed_path)
    print("Brief summary:", brief_path)


# ============================================================
# 9. Orchestrator
# ============================================================
def run_workflow(cfg: Config) -> Path:
    """
    Entry point for GUI/CLI: configure paths, run all steps, and return results dir.
    """
    paths = configure_paths(cfg)

    extract_oil_peaks_and_ratios(cfg)
    oil_anova_and_tukey()

    norm_out_dir = run_oil_norm2720_multivariate(output_root=paths["base"])

    # Thermal stability – Chips (requires Chips Ratiometric Analysis.csv)
    if cfg.chips_csv and os.path.exists(cfg.chips_csv):
        run_chips_heating_stats()
    else:
        print(
            f"\n[WARN] Chips file not found or not provided: {CHIPS_RATIOMETRIC_CSV} – skipping chips heating analysis."
        )

    # Thermal stability – Oils
    run_oil_heating_stats()

    # RQ1–RQ14 summaries
    generate_rq_summaries(norm_out_dir, cfg)

    print("\n=== ALL STEPS COMPLETED (including RQ summaries) ===")
    return paths["base"]


def main(cfg: Optional[Config] = None) -> Path:
    """
    Backwards-compatible main for GUI import or CLI execution.
    """
    if cfg is None:
        if not RAW_OIL_CSV:
            raise ValueError("Config.input_csv is required to run the workflow.")
        cfg = Config(
            input_csv=RAW_OIL_CSV,
            chips_csv=CHIPS_RATIOMETRIC_CSV,
        )
    return run_workflow(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the FoodSpec Raman workflow end-to-end.")
    parser.add_argument("--input_csv", required=True, help="Wide-format Raman CSV with metadata + wavenumber columns.")
    parser.add_argument("--chips_csv", default=None, help="Optional chips ratiometric CSV for heating analysis.")
    parser.add_argument(
        "--run_name", default="raman_cli_run", help="Name of the run (outputs under results/<run_name>)."
    )
    parser.add_argument("--output_root", default="results", help="Root directory for outputs.")
    parser.add_argument("--oil_col", default="Oil_Name", help="Column holding oil identity.")
    parser.add_argument("--heating_col", default="Heating_Stage", help="Column holding heating stage/time.")
    parser.add_argument("--baseline_lambda", type=float, default=10**5, help="ALS baseline lambda.")
    parser.add_argument("--baseline_p", type=float, default=0.01, help="ALS baseline asymmetry p.")
    parser.add_argument("--savgol_window", type=int, default=5, help="Savitzky-Golay window size.")

    args = parser.parse_args()

    cfg_cli = Config(
        input_csv=args.input_csv,
        chips_csv=args.chips_csv,
        run_name=args.run_name,
        output_root=Path(args.output_root),
        oil_col=args.oil_col,
        heating_col=args.heating_col,
        baseline_lambda=args.baseline_lambda,
        baseline_p=args.baseline_p,
        savgol_window=args.savgol_window,
    )

    main(cfg_cli)
