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

from foodspec.qc import Policy, apply_qc_policy


def test_policy_flag_uses_pass_column() -> None:
    qc = pd.DataFrame({"pass": [True, False], "fail_reasons": ["", "snr<3"]})
    mask, weights, summary = apply_qc_policy(qc, Policy(mode="flag"))

    assert mask.dtype == bool and weights.dtype == float
    assert mask.tolist() == [True, True]
    assert weights.tolist() == [1.0, 1.0]
    assert list(summary.columns) == ["pass", "action", "weight", "fail_reasons"]
    assert summary.loc[1, "fail_reasons"] == "snr<3"


def test_policy_drop_excludes_failures() -> None:
    qc = pd.DataFrame({"pass": [True, False]})
    mask, weights, summary = apply_qc_policy(qc, Policy(mode="drop"))

    assert mask.tolist() == [True, False]
    assert weights.tolist() == [1.0, 1.0]
    assert (summary["weight"] == 1.0).all()
    assert (summary["action"] == "drop").all()


def test_policy_downweight_applies_weight() -> None:
    qc = pd.DataFrame({"pass": [True, False]})
    mask, weights, summary = apply_qc_policy(qc, Policy(mode="downweight", fail_weight=0.3))

    assert mask.tolist() == [True, True]
    assert np.allclose(weights, [1.0, 0.3])
    assert summary.loc[1, "weight"] == pytest.approx(0.3)


def test_policy_thresholds_compute_pass_when_missing() -> None:
    qc = pd.DataFrame({"snr": [5.0, 2.0], "drift": [0.1, 0.5]})
    pol = Policy(mode="drop", thresholds={"snr": {"min": 3.0}, "drift": {"max": 0.3}})

    mask, weights, summary = apply_qc_policy(qc, pol)
    assert mask.tolist() == [True, False]
    # fail reasons should include drift
    assert "drift>0.3" in summary.loc[1, "fail_reasons"]


def test_policy_validation_errors() -> None:
    qc = pd.DataFrame({"snr": [5.0]})
    with pytest.raises(ValueError):
        apply_qc_policy(qc, Policy(mode="flag"))

    with pytest.raises(ValueError):
        apply_qc_policy(pd.DataFrame({"pass": [True]}), Policy(mode="invalid"))
