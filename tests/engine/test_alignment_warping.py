from __future__ import annotations

import numpy as np

from foodspec.data_objects.spectra_set import FoodSpectrumSet
from foodspec.engine.preprocessing.engine import AlignmentStep


def _make_ds():
    wavenumbers = np.linspace(1000, 1100, 50)
    base = np.sin(np.linspace(0, 3.14, 50))
    X = np.vstack([base, np.roll(base, 2), np.roll(base, -3)])
    return FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=None, modality="raman")


def test_alignment_piecewise():
    ds = _make_ds()
    step = AlignmentStep(method="piecewise", max_shift=3, segment_size=10)
    aligned = step.transform(ds)
    assert aligned.x.shape == ds.x.shape


def test_alignment_linear_warp():
    ds = _make_ds()
    step = AlignmentStep(method="linear_warp")
    aligned = step.transform(ds)
    assert aligned.x.shape == ds.x.shape
