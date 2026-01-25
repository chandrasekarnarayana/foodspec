"""Tree-based model helpers."""
from __future__ import annotations

from foodspec.chemometrics.models import make_classifier


def make_tree_classifier(model_name: str = "rf", **kwargs):
    """Create a tree-based classifier (rf, xgb, lgbm)."""

    return make_classifier(model_name, **kwargs)


__all__ = ["make_tree_classifier"]

