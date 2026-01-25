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

from foodspec.qc.base import QCMetric, QCSummary


class MeanMetric:
    """Simple metric returning per-sample mean."""

    def compute(self, X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"mean": X.mean(axis=1)})


def test_qcmetric_compute_shape() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    meta = pd.DataFrame({"sample_id": ["a", "b"]})
    metric: QCMetric = MeanMetric()

    df = metric.compute(X, meta)
    assert list(df.columns) == ["mean"]
    assert df.shape[0] == X.shape[0]


def test_qc_summary_pass_fail() -> None:
    metrics = pd.DataFrame({"snr": [5.0, 2.0], "drift": [0.1, 0.5]})
    summary = QCSummary({"snr": {"min": 3.0}, "drift": {"max": 0.3}})

    result = summary.evaluate(metrics)
    assert result.loc[0, "pass"] is True
    assert result.loc[1, "pass"] is False
    assert "drift>0.3" in result.loc[1, "fail_reasons"]


def test_qc_summary_missing_metric() -> None:
    metrics = pd.DataFrame({"snr": [5.0]})
    summary = QCSummary({"snr": {"min": 1.0}, "drift": {"max": 0.1}})

    with pytest.raises(ValueError):
        summary.evaluate(metrics)
