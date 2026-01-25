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

QC policy application: flag, drop, or downweight samples based on QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd

from foodspec.qc.base import QCSummary


@dataclass
class Policy:
    """QC policy specification.

    Parameters
    ----------
    mode : {"flag", "drop", "downweight"}
        Action to apply for failing samples.
    thresholds : Mapping[str, Mapping[str, float]], optional
        Metric thresholds to compute pass/fail if "pass" column not provided
        in the QC table. Uses the same schema as ``QCSummary`` (supports
        "min" and/or "max" per metric).
    fail_weight : float, default 0.5
        Weight to assign to failing samples when ``mode='downweight'``.

    Examples
    --------
    >>> import pandas as pd
    >>> qc = pd.DataFrame({"snr": [5.0, 2.0], "drift": [0.1, 0.5]})
    >>> pol = Policy(mode="drop", thresholds={"snr": {"min": 3.0}, "drift": {"max": 0.3}})
    >>> mask, weights, summary = apply_qc_policy(qc, pol)
    >>> mask.tolist()
    [True, False]
    >>> weights.tolist()
    [1.0, 1.0]
    >>> list(summary.columns)
    ['pass', 'action', 'weight', 'fail_reasons']
    """

    mode: str
    thresholds: Mapping[str, Mapping[str, float]] | None = None
    fail_weight: float = 0.5


def _ensure_pass_column(qc_table: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    """Ensure a 'pass' column exists; compute from thresholds if needed.

    Returns a copy of the table (does not modify input in-place).
    """

    if "pass" in qc_table.columns:
        return qc_table.copy()

    if not policy.thresholds:
        raise ValueError(
            "QC table missing 'pass' column and no thresholds provided in policy. "
            "Either include a 'pass' boolean in qc_table or set Policy.thresholds."
        )

    summary = QCSummary(policy.thresholds).evaluate(qc_table)
    out = qc_table.copy()
    out["pass"] = summary["pass"].to_numpy()
    out["fail_reasons"] = summary["fail_reasons"].astype(str).to_numpy()
    return out


def apply_qc_policy(
    qc_table: pd.DataFrame, policy: Policy
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply a QC policy to a QC table.

    Parameters
    ----------
    qc_table : DataFrame
        Per-sample QC table. Must contain a 'pass' boolean column, or
        policy.thresholds must be provided to derive pass/fail. If available,
        a 'fail_reasons' column will be forwarded to the summary.
    policy : Policy
        Policy specifying the action and optional thresholds/weights.

    Returns
    -------
    mask : ndarray of bool, shape (n_samples,)
        Inclusion mask for downstream steps (False entries are dropped).
    weights : ndarray of float, shape (n_samples,)
        Sample weights for downstream steps.
    summary : DataFrame
        Per-sample summary with columns: ['pass', 'action', 'weight', 'fail_reasons'].

    Raises
    ------
    ValueError
        If mode is invalid or pass/fail cannot be determined.
    """

    if policy.mode not in {"flag", "drop", "downweight"}:
        raise ValueError("Policy.mode must be one of: 'flag', 'drop', 'downweight'")

    table = _ensure_pass_column(qc_table, policy)
    n = len(table)
    pass_vec = table["pass"].astype(bool).to_numpy()

    if policy.mode == "flag":
        mask = np.ones(n, dtype=bool)
        weights = np.ones(n, dtype=float)
    elif policy.mode == "drop":
        mask = pass_vec.copy()
        weights = np.ones(n, dtype=float)
    else:  # downweight
        if not (0.0 <= policy.fail_weight <= 1.0):
            raise ValueError("fail_weight must be within [0, 1]")
        mask = np.ones(n, dtype=bool)
        weights = np.where(pass_vec, 1.0, float(policy.fail_weight))

    # Build per-sample summary
    summary: MutableMapping[str, object] = {
        "pass": pass_vec,
        "action": np.array([policy.mode] * n, dtype=object),
        "weight": weights,
    }
    if "fail_reasons" in table.columns:
        summary["fail_reasons"] = table["fail_reasons"].astype(str).to_numpy()
    else:
        summary["fail_reasons"] = np.array([""] * n, dtype=object)

    return mask, weights, pd.DataFrame(summary, index=table.index)


__all__ = ["Policy", "apply_qc_policy"]
