from __future__ import annotations
"""
NNLS mixture unmixing: solve x ≈ A·c with c ≥ 0 and compute reconstruction error
and bootstrap confidence intervals for concentrations.
"""


from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import nnls


@dataclass
class NNLSResult:
    concentrations: np.ndarray
    recon_error: float


def solve_nnls(A: np.ndarray, x: np.ndarray) -> NNLSResult:
    c, rnorm = nnls(A, x)
    recon = np.linalg.norm(A @ c - x)
    return NNLSResult(concentrations=c, recon_error=float(recon))


def bootstrap_nnls(A: np.ndarray, x: np.ndarray, n_boot: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        res = solve_nnls(A, xb)
        samples.append(res.concentrations)
    S = np.stack(samples)
    ci_low = np.percentile(S, 2.5, axis=0)
    ci_high = np.percentile(S, 97.5, axis=0)
    return ci_low, ci_high
