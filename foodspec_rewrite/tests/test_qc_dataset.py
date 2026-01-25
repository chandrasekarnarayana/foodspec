"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
"""

import numpy as np
import pandas as pd
import pytest

from foodspec.qc import DatasetQC


def test_batch_drift_detects_shift() -> None:
    rng = np.random.default_rng(0)
    batch_a = rng.normal(loc=0.0, scale=0.05, size=(5, 20))
    batch_b = rng.normal(loc=1.0, scale=0.05, size=(5, 20))

    X = np.vstack([batch_a, batch_b])
    meta = pd.DataFrame({"batch": ["A"] * 5 + ["B"] * 5})

    qc = DatasetQC()
    drift = qc.compute_batch_drift(X, meta)

    scores = dict(zip(drift["batch"], drift["drift_score"]))
    assert scores["A"] < scores["B"]
    assert drift.loc[0, "reference"] == "A"
    assert list(drift.columns) == ["batch", "n_samples", "reference", "drift_score"]


def test_replicate_consistency_variance() -> None:
    rng = np.random.default_rng(1)
    base = np.linspace(0, 1, 15)
    rep1 = np.vstack([base, base + 0.001, base + 0.002])
    rep2 = np.vstack([base + rng.normal(0, 0.2, size=base.shape) for _ in range(3)])

    X = np.vstack([rep1, rep2])
    meta = pd.DataFrame({"replicate_id": ["r1"] * 3 + ["r2"] * 3})

    qc = DatasetQC()
    consistency = qc.compute_replicate_consistency(X, meta)

    var_map = dict(zip(consistency["replicate_id"], consistency["within_variance"]))
    assert var_map["r2"] > var_map["r1"]
    assert consistency.loc[0, "n_samples"] == 3


def test_dataset_qc_combined_and_missing_columns() -> None:
    X = np.array([[1.0, 2.0], [1.1, 2.1]])
    meta = pd.DataFrame({"batch": ["A", "A"], "replicate_id": ["r1", "r1"]})

    qc = DatasetQC()
    result = qc.compute(X, meta)

    assert set(result.keys()) == {"batch_drift", "replicate_consistency"}
    assert result["batch_drift"].shape[0] == 1
    assert result["replicate_consistency"].shape[0] == 1

    with pytest.raises(ValueError):
        qc.compute_batch_drift(X, pd.DataFrame({"replicate_id": ["r1", "r1"]}))

    with pytest.raises(ValueError):
        qc.compute_replicate_consistency(X, pd.DataFrame({"batch": ["A", "A"]}))
