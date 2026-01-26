import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from foodspec.output_bundle import save_qc_artifacts
from foodspec.qc.multivariate import (
    MultivariateQCPolicy,
    batch_drift,
    compute_pca_outlier_scores,
    hotelling_t2,
    outlier_flags,
)


def test_multivariate_outliers_detect_mad_strategy():
    rng = np.random.default_rng(42)
    baseline = rng.normal(size=(40, 3))
    # Inject deterministic outliers
    baseline[:2] += 12.0

    res = compute_pca_outlier_scores(baseline, method="robust")
    policy = MultivariateQCPolicy(action="flag", severity="warn", threshold_strategy="mad", mad_multiplier=4.0)
    thresholds = res["thresholds"].copy()
    thresholds["score_distance"] = float(
        np.median(res["score_distance"]) + policy.mad_multiplier * stats.median_abs_deviation(res["score_distance"], scale=1.0)
    )
    thresholds["orthogonal_distance"] = float(
        np.median(res["orthogonal_distance"]) + policy.mad_multiplier * stats.median_abs_deviation(res["orthogonal_distance"], scale=1.0)
    )

    table = outlier_flags(
        sample_ids=list(range(len(baseline))),
        scores={
            "score_distance": np.asarray(res["score_distance"], dtype=float),
            "orthogonal_distance": np.asarray(res["orthogonal_distance"], dtype=float),
        },
        thresholds=thresholds,
        policy=policy,
    )

    assert table["flag"].sum() >= 2
    assert set(table.loc[table["flag"], "reason_code"]) <= {"MV_OUTLIER_T2", "MV_OUTLIER_OD"}


def test_hotelling_limits_and_flags():
    rng = np.random.default_rng(123)
    scores = rng.normal(size=(50, 4))
    t2_vals, limit = hotelling_t2(scores, covariance="empirical", alpha=0.05)
    policy = MultivariateQCPolicy(action="flag", threshold_strategy="chi2")
    thresholds = {"t2": limit}
    table = outlier_flags(list(range(len(scores))), {"t2": t2_vals}, thresholds, policy)
    assert 0 < limit
    assert table["flag"].sum() > 0  # tail events


def test_batch_drift_and_artifact_writes(tmp_path: Path):
    scores = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 3.0])
    batches = ["A"] * 5 + ["B"] * 5
    drift_df = batch_drift(scores, batches, warn_threshold=1.0, fail_threshold=2.0)
    assert (drift_df["status"] == "fail").any()

    outliers_df = pd.DataFrame(
        [
            {"sample_id": i, "method": "hotelling_t2", "score": float(i), "threshold": 2.5, "flag": i > 3, "reason_code": "MV_OUTLIER_T2", "severity": "warn", "action": "flag"}
            for i in range(6)
        ]
    )
    qc_summary = {"multivariate": {"status": "warn", "outliers": {"n_flagged": 2}}}

    save_qc_artifacts(
        tmp_path,
        {
            "multivariate_outliers": outliers_df,
            "multivariate_drift": drift_df,
            "qc_summary": qc_summary,
        },
    )

    outliers_path = tmp_path / "qc" / "multivariate_outliers.csv"
    drift_path = tmp_path / "qc" / "multivariate_drift.csv"
    summary_path = tmp_path / "qc" / "qc_summary.json"

    assert outliers_path.exists()
    assert drift_path.exists()
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["multivariate"]["status"] == "warn"
