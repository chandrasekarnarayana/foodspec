"""Classical model factories (shim to foodspec.chemometrics.models)."""

from __future__ import annotations

from foodspec.chemometrics.models import make_classifier, make_regressor

__all__ = ["make_classifier", "make_regressor"]
