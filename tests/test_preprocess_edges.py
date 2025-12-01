import numpy as np
import pytest

from foodspec.preprocess.baseline import ALSBaseline, PolynomialBaseline, RubberbandBaseline
from foodspec.preprocess.normalization import VectorNormalizer, SNVNormalizer, MSCNormalizer
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother, MovingAverageSmoother


def test_baseline_invalid_shapes():
    als = ALSBaseline()
    with pytest.raises(ValueError):
        als.transform(np.ones(5))
    poly = PolynomialBaseline(degree=3)
    with pytest.raises(ValueError):
        poly.transform(np.ones(5))


def test_rubberband_baseline_runs():
    x = np.linspace(0, 10, 20)
    spectrum = np.sin(x) + np.linspace(0, 1, 20)
    rb = RubberbandBaseline()
    corrected = rb.transform(spectrum.reshape(1, -1))
    assert corrected.shape == (1, spectrum.shape[0])


def test_normalization_zero_norm_and_snv_msc():
    X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    norm = VectorNormalizer(norm="l2")
    out = norm.transform(X)
    assert np.allclose(out[1], X[1] / np.linalg.norm(X[1]))
    snv = SNVNormalizer()
    snv_out = snv.transform(X)
    assert np.allclose(np.mean(snv_out[1]), 0)
    msc = MSCNormalizer()
    msc.fit(X)
    msc_out = msc.transform(X)
    assert msc_out.shape == X.shape
    with pytest.raises(RuntimeError):
        MSCNormalizer().transform(X)


def test_smoothing_invalid_window_and_success():
    X = np.arange(15, dtype=float).reshape(1, -1)
    with pytest.raises(ValueError):
        SavitzkyGolaySmoother(window_length=4, polyorder=2).transform(X)
    sg = SavitzkyGolaySmoother(window_length=5, polyorder=2)
    sg_out = sg.transform(X)
    assert sg_out.shape == X.shape
    ma = MovingAverageSmoother(window_size=3)
    ma_out = ma.transform(X)
    assert ma_out.shape == X.shape
