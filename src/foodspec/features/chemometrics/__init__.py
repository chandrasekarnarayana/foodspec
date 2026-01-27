"""Chemometrics helpers under features namespace."""
from __future__ import annotations

from foodspec.modeling.validation.metrics import compute_vip_scores as compute_vip

from .pca import PCAResult, run_pca
from .pls import fit_pls_model

__all__ = ["PCAResult", "run_pca", "fit_pls_model", "compute_vip"]
