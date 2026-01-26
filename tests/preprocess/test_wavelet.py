from __future__ import annotations

import numpy as np
import pytest

from foodspec.preprocess.smoothing import WaveletDenoiser


def test_wavelet_denoise_shape():
    pytest.importorskip("pywt")
    X = np.random.RandomState(0).randn(4, 64)
    denoised = WaveletDenoiser(wavelet="db2", level=2).fit_transform(X)
    assert denoised.shape == X.shape
