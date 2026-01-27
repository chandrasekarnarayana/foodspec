"""ROC/AUC artifact saving and management utilities."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def save_roc_artifacts(
    output_dir: Path | str,
    roc_result: Any,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list,
) -> Dict[str, str]:
    """Save ROC diagnostics artifacts to disk.

    Parameters
    ----------
    output_dir : Path or str
        Directory to save artifacts in.
    roc_result : RocDiagnosticsResult
        Result from compute_roc_diagnostics().
    y_true : np.ndarray
        True labels.
    y_proba : np.ndarray
        Predicted probabilities.
    classes : list
        Class labels.

    Returns
    -------
    dict
        Dictionary mapping artifact type to file paths saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # 1. Save ROC summary CSV (per-class + micro/macro AUC + CI)
    roc_summary = _build_roc_summary_df(roc_result, classes)
    roc_summary_path = output_dir / "tables" / "roc_summary.csv"
    roc_summary_path.parent.mkdir(parents=True, exist_ok=True)
    roc_summary.to_csv(roc_summary_path, index=False)
    artifacts["roc_summary"] = str(roc_summary_path)

    # 2. Save thresholds CSV
    thresholds_df = _build_thresholds_df(roc_result, classes)
    if thresholds_df is not None:
        thresholds_path = output_dir / "tables" / "roc_thresholds.csv"
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds_df.to_csv(thresholds_path, index=False)
        artifacts["thresholds"] = str(thresholds_path)

    # 3. Save full diagnostics as JSON
    roc_json = _serialize_roc_result(roc_result)
    json_path = output_dir / "json" / "roc_diagnostics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(roc_json, f, indent=2)
    artifacts["roc_diagnostics_json"] = str(json_path)

    # 4. Plot ROC curves if matplotlib available
    if plt is not None:
        try:
            fig_paths = _plot_roc_curves(
                output_dir / "figures", roc_result, y_true, y_proba, classes
            )
            artifacts.update(fig_paths)
        except Exception as e:
            warnings.warn(f"Failed to create ROC plots: {e}", stacklevel=2)

    return artifacts


def _build_roc_summary_df(roc_result: Any, classes: list) -> Any:
    """Build ROC summary DataFrame."""
    import pandas as pd

    rows = []

    # Per-class metrics
    for class_label, metrics in roc_result.per_class.items():
        rows.append({
            "class": str(class_label),
            "type": "per_class",
            "auc": metrics.auc,
            "ci_lower": metrics.ci_lower or np.nan,
            "ci_upper": metrics.ci_upper or np.nan,
            "n_positives": metrics.n_positives or np.nan,
            "n_negatives": metrics.n_negatives or np.nan,
        })

    # Micro-average (multiclass only)
    if roc_result.micro is not None:
        rows.append({
            "class": "micro",
            "type": "aggregate",
            "auc": roc_result.micro.auc,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_positives": np.nan,
            "n_negatives": np.nan,
        })

    # Macro-average (multiclass only)
    if roc_result.macro_auc is not None:
        rows.append({
            "class": "macro",
            "type": "aggregate",
            "auc": roc_result.macro_auc,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_positives": np.nan,
            "n_negatives": np.nan,
        })

    return pd.DataFrame(rows)


def _build_thresholds_df(roc_result: Any, classes: list) -> Any:
    """Build thresholds DataFrame."""
    import pandas as pd

    if not roc_result.optimal_thresholds:
        return None

    rows = []
    for policy_name, threshold_result in roc_result.optimal_thresholds.items():
        rows.append({
            "policy": policy_name,
            "threshold": threshold_result.threshold,
            "sensitivity": threshold_result.sensitivity,
            "specificity": threshold_result.specificity,
            "ppv": threshold_result.ppv or np.nan,
            "npv": threshold_result.npv or np.nan,
            "j_statistic": threshold_result.j_statistic or np.nan,
        })

    return pd.DataFrame(rows) if rows else None


def _serialize_roc_result(roc_result: Any) -> Dict[str, Any]:
    """Serialize RocDiagnosticsResult to JSON-compatible dict."""
    result = {
        "metadata": roc_result.metadata,
        "per_class": {},
        "micro": None,
        "macro_auc": roc_result.macro_auc,
        "optimal_thresholds": {},
    }

    # Per-class metrics
    for class_label, metrics in roc_result.per_class.items():
        result["per_class"][str(class_label)] = {
            "auc": float(metrics.auc),
            "ci_lower": float(metrics.ci_lower) if metrics.ci_lower is not None else None,
            "ci_upper": float(metrics.ci_upper) if metrics.ci_upper is not None else None,
            "n_positives": int(metrics.n_positives) if metrics.n_positives is not None else None,
            "n_negatives": int(metrics.n_negatives) if metrics.n_negatives is not None else None,
        }

    # Micro-average
    if roc_result.micro is not None:
        result["micro"] = {
            "auc": float(roc_result.micro.auc),
            "n_samples": int(sum(m.n_positives + m.n_negatives for m in roc_result.per_class.values())),
        }

    # Optimal thresholds
    for policy_name, thr_result in roc_result.optimal_thresholds.items():
        result["optimal_thresholds"][policy_name] = {
            "threshold": float(thr_result.threshold),
            "sensitivity": float(thr_result.sensitivity),
            "specificity": float(thr_result.specificity),
            "ppv": float(thr_result.ppv) if thr_result.ppv is not None else None,
            "npv": float(thr_result.npv) if thr_result.npv is not None else None,
            "j_statistic": float(thr_result.j_statistic) if thr_result.j_statistic is not None else None,
        }

    return result


def _plot_roc_curves(
    figures_dir: Path,
    roc_result: Any,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list,
) -> Dict[str, str]:
    """Create and save ROC curve plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    n_classes = len(roc_result.per_class)

    # Determine if binary or multiclass
    if n_classes == 2:
        # Binary: single ROC curve
        class_label = list(roc_result.per_class.keys())[0]
        metrics = roc_result.per_class[class_label]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(metrics.fpr, metrics.tpr, lw=2, label=f"ROC (AUC={metrics.auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Binary Classification ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        roc_path = figures_dir / "roc_curve_binary.png"
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        artifacts["roc_curve_binary"] = str(roc_path)

    else:
        # Multiclass: per-class OvR curves
        n_rows = (n_classes + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
        axes = axes.flatten()

        for idx, (class_label, metrics) in enumerate(roc_result.per_class.items()):
            ax = axes[idx]
            ax.plot(metrics.fpr, metrics.tpr, lw=2, label=f"Class {class_label} (AUC={metrics.auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"OvR ROC: Class {class_label}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        roc_path = figures_dir / "roc_curve_per_class.png"
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        artifacts["roc_curve_per_class"] = str(roc_path)

        # Micro-average plot
        if roc_result.micro is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                roc_result.micro.fpr,
                roc_result.micro.tpr,
                lw=2,
                label=f"Micro-avg (AUC={roc_result.micro.auc:.3f})",
            )
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Multiclass Micro-Average ROC Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)

            micro_path = figures_dir / "roc_curve_micro.png"
            fig.savefig(micro_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            artifacts["roc_curve_micro"] = str(micro_path)

    return artifacts
