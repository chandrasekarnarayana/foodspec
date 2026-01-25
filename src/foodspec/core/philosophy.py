"""FoodSpec design philosophy, goals, and mindmap mapping."""
from __future__ import annotations

from typing import List, Dict

GOALS: List[str] = [
    "Protocol-driven spectroscopy workflows",
    "Reproducibility by default",
    "Trust and uncertainty as first-class outputs",
    "QC is mandatory, not optional",
    "Designed for food matrices with complex backgrounds",
]

NON_GOALS: List[str] = [
    "Not a general deep learning framework",
    "Not a vendor replacement tool",
    "Not claiming clinical or regulatory approval",
]

FEATURE_MAP: List[Dict[str, str]] = [
    {
        "mindmap_node": "Data Objects",
        "module": "foodspec.data_objects",
        "public_api": "Spectrum, SpectraSet, SpectralDataset",
        "cli": "foodspec io validate",
        "artifacts": "protocol.yaml, run_summary.json",
    },
    {
        "mindmap_node": "Data Extraction",
        "module": "foodspec.io",
        "public_api": "read_spectra, detect_format",
        "cli": "foodspec io validate",
        "artifacts": "ingest logs",
    },
    {
        "mindmap_node": "Programming Engine",
        "module": "foodspec.engine",
        "public_api": "preprocessing pipeline",
        "cli": "foodspec preprocess run",
        "artifacts": "preprocess logs",
    },
    {
        "mindmap_node": "Quality Control",
        "module": "foodspec.qc",
        "public_api": "qc engine, dataset_qc",
        "cli": "foodspec qc spectral|dataset",
        "artifacts": "qc_results.json",
    },
    {
        "mindmap_node": "Feature Engineering",
        "module": "foodspec.features",
        "public_api": "peaks, ratios, chemometrics",
        "cli": "foodspec features extract",
        "artifacts": "features tables",
    },
    {
        "mindmap_node": "Modeling & Validation",
        "module": "foodspec.modeling",
        "public_api": "training and validation",
        "cli": "foodspec train",
        "artifacts": "metrics.json",
    },
    {
        "mindmap_node": "Trust & Uncertainty",
        "module": "foodspec.trust",
        "public_api": "calibration, conformal",
        "cli": "foodspec report",
        "artifacts": "uncertainty_metrics.json",
    },
    {
        "mindmap_node": "Visualization & Reporting",
        "module": "foodspec.viz, foodspec.reporting",
        "public_api": "figures, report builders",
        "cli": "foodspec report",
        "artifacts": "html/pdf reports",
    },
]


def feature_map() -> List[Dict[str, str]]:
    """Return the mindmap-to-module mapping table."""

    return list(FEATURE_MAP)

