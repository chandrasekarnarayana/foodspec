import numpy as np

from foodspec.preprocess.ftir import AtmosphericCorrector, SimpleATRCorrector


def test_atmospheric_corrector_reduces_component():
    wn = np.linspace(1800, 2400, 400)
    base = np.sin(wn / 200) * 0.1
    water = np.exp(-0.5 * ((wn - 1900) / 30) ** 2)
    co2 = np.exp(-0.5 * ((wn - 2350) / 30) ** 2)
    spectrum = base + 0.8 * water + 0.6 * co2
    X = np.vstack([spectrum, spectrum * 1.05])

    corr = AtmosphericCorrector(alpha_water=1.0, alpha_co2=1.0)
    corr.fit(X, wavenumbers=wn)
    X_corr = corr.transform(X)

    # Project corrected spectra back onto atmospheric bases
    bases = corr._bases
    proj = X_corr @ bases
    orig_proj = X @ bases
    assert np.linalg.norm(proj) < np.linalg.norm(orig_proj) * 0.5


def test_simple_atr_outputs_finite():
    wn = np.linspace(500, 4000, 300)
    X = np.vstack([np.cos(wn / 500), np.sin(wn / 400)])
    atr = SimpleATRCorrector()
    atr.fit(X, wavenumbers=wn)
    X_corr = atr.transform(X)
    assert X_corr.shape == X.shape
    assert np.isfinite(X_corr).all()
