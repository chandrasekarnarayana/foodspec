"""Reusable figure caption generator for papers and slide decks."""

from __future__ import annotations

from typing import List, Literal, Optional


def panel_labels(count: int) -> List[str]:
    """Return panel labels A, B, C... up to count."""

    labels = []
    for i in range(count):
        labels.append(chr(ord("A") + i))
    return labels


def _join_highlights(highlights: Optional[List[str]]) -> str:
    if not highlights:
        return ""
    if len(highlights) == 1:
        return highlights[0]
    return "; ".join(highlights)


def generate_caption(
    figure_type: Literal[
        "roc_curve",
        "confusion_matrix",
        "calibration_curve",
        "drift_dashboard",
        "feature_importance",
    ],
    dataset: str,
    highlights: Optional[List[str]] = None,
    stats: Optional[str] = None,
    modality: Optional[str] = None,
    panels: Optional[List[str]] = None,
) -> str:
    """Generate a compact caption with consistent structure.

    Parameters
    ----------
    figure_type : str
        One of the supported templates.
    dataset : str
        Dataset or cohort name.
    highlights : list of str, optional
        Key findings to weave into the caption.
    stats : str, optional
        Statistical summary to append (e.g., "AUC=0.92, 95% CI 0.88-0.96").
    modality : str, optional
        Modality label to mention.
    panels : list of str, optional
        Panel labels if using a multi-panel figure.
    """

    mod = f" using {modality}" if modality else ""
    highlight_text = _join_highlights(highlights)
    panel_text = f" Panels {', '.join(panels)}." if panels else ""

    templates = {
        "roc_curve": (
            "Receiver operating characteristic for {dataset}{mod}. "
            "Curves show mean performance with shaded 95% confidence. {highlight_text}{stats}{panel_text}"
        ),
        "confusion_matrix": (
            "Confusion matrix for {dataset}{mod}. "
            "Diagonal cells indicate correct predictions; off-diagonal cells highlight systematic errors. {highlight_text}{stats}{panel_text}"
        ),
        "calibration_curve": (
            "Calibration reliability plot for {dataset}{mod}. "
            "Perfect calibration aligns with the diagonal; deviation indicates miscalibration. {highlight_text}{stats}{panel_text}"
        ),
        "drift_dashboard": (
            "Production drift dashboard for {dataset}{mod}. "
            "Population stability index and KL divergence track distribution shift over time. {highlight_text}{stats}{panel_text}"
        ),
        "feature_importance": (
            "Feature importance summary for {dataset}{mod}. "
            "Higher values indicate stronger contribution to the model decision boundary. {highlight_text}{stats}{panel_text}"
        ),
    }

    if figure_type not in templates:
        raise ValueError(f"Unsupported figure_type: {figure_type}")

    stats_text = f" {stats}" if stats else ""
    caption = templates[figure_type].format(
        dataset=dataset,
        mod=mod,
        highlight_text=(highlight_text + " ") if highlight_text else "",
        stats=stats_text,
        panel_text=panel_text,
    )
    return " ".join(caption.split())
