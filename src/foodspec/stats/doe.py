from __future__ import annotations
"""
Design of experiments helpers (factorial, fractional, response surface, D-optimal).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def full_factorial_2level(factors: Sequence[str]) -> pd.DataFrame:
    """Generate a full 2-level factorial design (-1/+1)."""
    n = len(factors)
    grid = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)
    return pd.DataFrame(grid, columns=list(factors))


def fractional_factorial_2level(
    base_factors: Sequence[str],
    generators: Dict[str, str],
) -> pd.DataFrame:
    """Generate a fractional factorial design from generators.

    generators example: {"D": "AB", "E": "AC"} means D = A*B, E = A*C.
    """
    base = full_factorial_2level(base_factors)
    for new_factor, formula in generators.items():
        cols = [c.strip() for c in formula]
        values = np.ones(len(base))
        for col in cols:
            if col not in base.columns:
                raise ValueError(f"Generator references unknown factor '{col}'.")
            values *= base[col].to_numpy()
        base[new_factor] = values
    return base


def central_composite_design(
    factors: Sequence[str],
    *,
    alpha: float = 1.414,
) -> pd.DataFrame:
    """Generate a central composite design for response surface modeling."""
    k = len(factors)
    factorial = full_factorial_2level(factors)
    center = pd.DataFrame([[0.0] * k], columns=factors)
    star = []
    for i in range(k):
        for sign in (-1, 1):
            row = [0.0] * k
            row[i] = sign * alpha
            star.append(row)
    star_df = pd.DataFrame(star, columns=factors)
    return pd.concat([factorial, star_df, center], ignore_index=True)


def randomized_block_design(
    treatments: Sequence[str],
    blocks: Sequence[str],
    *,
    random_state: int = 0,
) -> pd.DataFrame:
    """Generate a randomized block design (treatments x blocks)."""
    rng = np.random.default_rng(random_state)
    rows = []
    for block in blocks:
        block_rows = [{"block": block, "treatment": t} for t in treatments]
        rng.shuffle(block_rows)
        rows.extend(block_rows)
    return pd.DataFrame(rows)


@dataclass
class DOptimalResult:
    design: pd.DataFrame
    determinant: float


def d_optimal_design(
    candidate_matrix: np.ndarray,
    n_runs: int,
    *,
    random_state: int = 0,
) -> DOptimalResult:
    """Greedy D-optimal design from candidate rows."""
    X = np.asarray(candidate_matrix, dtype=float)
    if n_runs > X.shape[0]:
        raise ValueError("n_runs cannot exceed number of candidates.")
    rng = np.random.default_rng(random_state)
    remaining = list(range(X.shape[0]))
    selected: List[int] = []

    while len(selected) < n_runs:
        best_det = -np.inf
        best_idx = None
        for idx in remaining:
            trial = selected + [idx]
            XtX = X[trial].T @ X[trial]
            det = np.linalg.det(XtX + np.eye(XtX.shape[0]) * 1e-9)
            if det > best_det:
                best_det = det
                best_idx = idx
        if best_idx is None:
            best_idx = rng.choice(remaining)
        selected.append(best_idx)
        remaining.remove(best_idx)

    design = pd.DataFrame(X[selected])
    det_final = float(np.linalg.det(X[selected].T @ X[selected] + np.eye(X.shape[1]) * 1e-9))
    return DOptimalResult(design=design, determinant=det_final)


__all__ = [
    "full_factorial_2level",
    "fractional_factorial_2level",
    "central_composite_design",
    "randomized_block_design",
    "DOptimalResult",
    "d_optimal_design",
]
