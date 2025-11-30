"""Protocol-level benchmarks on public datasets."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

from foodspec.data.public import load_public_evoo_sunflower_raman, load_public_mendeley_oils
from foodspec.chemometrics.models import make_classifier
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.normalization import VectorNormalizer, MSCNormalizer
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.reporting import create_run_dir, write_json, write_markdown_report, summarize_metrics_for_markdown

__all__ = ["run_protocol_benchmarks"]


def _classification_benchmark(random_state: int = 42) -> Tuple[Dict, pd.DataFrame]:
    ds = load_public_mendeley_oils()
    X = ds.x
    y = ds.metadata["oil_type"].to_numpy()
    pipe = Pipeline(
        [
            ("als", ALSBaseline(lambda_=1e5, p=0.01, max_iter=10)),
            ("savgol", SavitzkyGolaySmoother(window_length=11, polyorder=3)),
            ("norm", MSCNormalizer()),
            ("clf", make_classifier("rf", random_state=random_state)),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    cm = confusion_matrix(y_test, preds, labels=np.unique(y))
    metrics = {
        "task": "oil_classification",
        "accuracy": acc,
        "f1_weighted": f1,
        "labels": np.unique(y).tolist(),
        "confusion_matrix": cm.tolist(),
    }
    cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
    return metrics, cm_df


def _mixture_benchmark(random_state: int = 42) -> Dict:
    ds = load_public_evoo_sunflower_raman()
    X = ds.x
    y = ds.metadata["mixture_fraction_evoo"].to_numpy()
    # Simple PLS regression
    pls = PLSRegression(n_components=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    pls.fit(X_train, y_train)
    preds = pls.predict(X_test).ravel()
    r2 = r2_score(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    return {
        "task": "mixture_regression",
        "r2": float(r2),
        "rmse": rmse,
    }


def run_protocol_benchmarks(
    output_dir: PathLike,
    random_state: int = 42,
) -> dict:
    """
    Run core protocol benchmarks on public datasets and save reports.

    Returns a dict summarizing metrics.
    """

    out_base = Path(output_dir)
    run_dir = create_run_dir(out_base, "protocol")
    summary: Dict[str, Dict] = {}

    # Classification benchmark
    try:
        metrics_cls, cm_df = _classification_benchmark(random_state=random_state)
        summary["classification"] = metrics_cls
        write_json(run_dir / "classification_metrics.json", metrics_cls)
        cm_df.to_csv(run_dir / "classification_confusion_matrix.csv")
    except FileNotFoundError as exc:
        summary["classification_error"] = str(exc)

    # Mixture benchmark
    try:
        metrics_mix = _mixture_benchmark(random_state=random_state)
        summary["mixture"] = metrics_mix
        write_json(run_dir / "mixture_metrics.json", metrics_mix)
    except FileNotFoundError as exc:
        summary["mixture_error"] = str(exc)

    # Report
    report_sections = {
        "Overview": "Protocol benchmarks on public datasets (oil classification, mixture regression).",
        "Metrics": summarize_metrics_for_markdown(summary),
    }
    write_markdown_report(run_dir / "report.md", title="Protocol Benchmarks", sections=report_sections)
    summary["run_dir"] = str(run_dir)
    return summary
