import numpy as np

from foodspec.chemometrics.mixture import nnls_mixture, mcr_als


def test_nnls_mixture_two_components():
    s1 = np.array([1.0, 0.0, 0.0])
    s2 = np.array([0.0, 1.0, 0.0])
    pure = np.vstack([s1, s2]).T  # shape (3, 2)
    mixture = 0.7 * s1 + 0.3 * s2
    coeffs, res = nnls_mixture(mixture, pure)
    assert np.allclose(coeffs, [0.7, 0.3], atol=1e-2)
    assert res < 1e-6


def test_mcr_als_simple_mixture():
    s1 = np.array([1.0, 2.0])
    s2 = np.array([0.5, 1.0])
    pure = np.vstack([s1, s2]).T
    mixtures = np.array(
        [
            0.2 * s1 + 0.8 * s2,
            0.5 * s1 + 0.5 * s2,
            0.8 * s1 + 0.2 * s2,
        ]
    )
    C, S = mcr_als(mixtures, n_components=2, max_iter=200, tol=1e-7, random_state=0)
    recon = C @ S.T
    err = np.linalg.norm(mixtures - recon) / np.linalg.norm(mixtures)
    assert err < 1e-2
    # Component correlation check
    corr0_s1 = np.corrcoef(S[:, 0], s1)[0, 1]
    corr1_s2 = np.corrcoef(S[:, 1], s2)[0, 1]
    assert corr0_s1 > 0.9 or corr1_s2 > 0.9
