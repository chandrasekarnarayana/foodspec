"""Model factories."""

from __future__ import annotations

from .classical import make_classifier, make_regressor
from .trees import make_tree_classifier

__all__ = ["make_classifier", "make_regressor", "make_tree_classifier"]
