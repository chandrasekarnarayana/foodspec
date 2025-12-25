import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA

from foodspec.chemometrics.models import make_pls_da
from foodspec.chemometrics.validation import (
    compute_explained_variance,
    compute_q2_rmsec_rmsep,
    compute_vip_scores,
    hotelling_t2_q_residuals,
    make_pca_report,
    permutation_pls_da,
    vip_table_with_meanings,
)


def test_pca_metrics_and_report():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    pca = PCA(n_components=3).fit(X)
    scores = pca.transform(X)
    report = make_pca_report(pca, scores, pca.components_.T)
    ev = compute_explained_variance(pca)
    t2, qres = hotelling_t2_q_residuals(pca, X)
    assert report["scores"].shape[1] == 3
    assert "explained_variance_ratio" in report
    assert ev["explained_variance_total"] > 0
    assert t2.shape[0] == X.shape[0]
    assert qres.min() >= 0


def test_q2_rmsec_rmsep_and_vip():
    X, y = make_regression(n_samples=60, n_features=6, noise=0.1, random_state=1)
    pls = PLSRegression(n_components=3).fit(X, y)
    q_metrics = compute_q2_rmsec_rmsep(pls, X, y, X, y)
    assert q_metrics["rmsec"] > 0
    vip = compute_vip_scores(pls, X, y)
    assert vip.shape[0] == X.shape[1]


def test_permutation_pls_da_and_vip_table():
    X, y = make_classification(n_samples=50, n_features=6, n_informative=4, random_state=2)
    pls_da = make_pls_da(n_components=2)
    score, perm_scores, pval = permutation_pls_da(pls_da, X, y, n_permutations=5, random_state=0)
    assert perm_scores.shape[0] == 5
    wn = np.linspace(1000, 1100, num=6)
    vip = np.linspace(1, 2, num=6)
    table = vip_table_with_meanings(vip, wn, top_n=3, modality="raman")
    assert len(table) == 3
    assert {"wavenumber", "vip", "meaning"}.issubset(table.columns)
