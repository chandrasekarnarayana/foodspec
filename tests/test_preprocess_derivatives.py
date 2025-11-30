import numpy as np

from foodspec.preprocess.derivatives import DerivativeTransformer


def test_derivative_transformer_smoke():
    wavenumbers = np.linspace(800, 1200, 101)
    peak = np.exp(-0.5 * ((wavenumbers - 1000) / 20) ** 2)
    ramp = np.linspace(0, 1, wavenumbers.size)
    spectrum = ramp + peak
    X = np.vstack([spectrum, spectrum * 1.1])

    dt = DerivativeTransformer(order=1, window_length=11, polyorder=3)
    X_der = dt.fit_transform(X)

    assert X_der.shape == X.shape
    assert np.isfinite(X_der).all()

