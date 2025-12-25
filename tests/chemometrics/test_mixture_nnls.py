import numpy as np

from foodspec.chemometrics.mixture_nnls import bootstrap_nnls, solve_nnls


def test_nnls_and_bootstrap_ci():
    # Two basis spectra; mixture is known combination
    A = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)  # 3 points, 2 components
    c_true = np.array([0.3, 0.7], dtype=float)
    x = A @ c_true
    res = solve_nnls(A, x)
    assert np.allclose(res.concentrations, c_true, atol=1e-6)
    low, high = bootstrap_nnls(A, x, n_boot=50)
    assert low.shape == c_true.shape
    assert high.shape == c_true.shape
