import numpy as np

from foodspec.chemometrics.models import make_simca
from foodspec.chemometrics.mixture import mcr_als


def test_simca_separates_simple_clusters():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0.0, 0.25, size=(25, 6))
    X1 = rng.normal(2.0, 0.25, size=(25, 6))
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))

    model = make_simca(n_components=3, alpha=0.99)
    model.fit(X, y)
    preds = model.predict(X)
    acc = float((preds == y).mean())
    assert acc > 0.9

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_mcr_als_reconstruction_error_small():
    rng = np.random.default_rng(1)
    # Create synthetic pure spectra (n_points=8, n_components=2)
    S_true = np.array(
        [
            [1.0, 0.2],
            [0.8, 0.3],
            [0.6, 0.5],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.9],
            [0.1, 1.0],
            [0.05, 0.8],
        ]
    )
    C_true = rng.uniform(0.1, 1.0, size=(30, 2))
    mixtures = C_true @ S_true.T
    mixtures += rng.normal(0, 0.01, size=mixtures.shape)

    C_est, S_est = mcr_als(mixtures, n_components=2, max_iter=50, random_state=0)
    recon = C_est @ S_est.T
    rel_err = np.linalg.norm(mixtures - recon) / np.linalg.norm(mixtures)
    assert rel_err < 0.25
