import numpy as np
from sklearn.pipeline import Pipeline

from foodspec.preprocess.ftir import AtmosphericCorrector, SimpleATRCorrector


def test_ftir_pipeline_with_wavenumber_mixin():
    wn = np.linspace(1800, 2400, 200)
    X = np.vstack([np.sin(wn / 300), np.cos(wn / 250)])

    atm = AtmosphericCorrector().set_wavenumbers(wn)
    atr = SimpleATRCorrector().set_wavenumbers(wn)

    pipe = Pipeline([("atm", atm), ("atr", atr)])
    X_out = pipe.fit_transform(X)
    assert X_out.shape == X.shape
    assert np.isfinite(X_out).all()
