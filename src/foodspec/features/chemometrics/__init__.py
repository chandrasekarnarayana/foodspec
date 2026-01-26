"""Chemometrics helpers under features namespace."""
from __future__ import annotations

from .pca import PCAResult, run_pca
from .pls import fit_pls_model
from foodspec.modeling.validation.metrics import compute_vip_scores as compute_vip

__all__ = ["PCAResult", "run_pca", "fit_pls_model", "compute_vip"]
