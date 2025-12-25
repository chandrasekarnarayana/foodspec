import numpy as np

from foodspec.preprocess.normalization import MSCNormalizer, SNVNormalizer


def test_snv_normalizer_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(5, 10))
    snv = SNVNormalizer()
    Xn = snv.fit_transform(X)
    means = Xn.mean(axis=1)
    stds = Xn.std(axis=1)
    assert np.allclose(means, 0, atol=1e-8)
    assert np.allclose(stds, 1, atol=1e-8)


def test_msc_normalizer_reference_preserved():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=(1, 8))
    X = np.vstack([ref, ref * 1.2 + 0.5])
    msc = MSCNormalizer()
    msc.fit(X)
    Xc = msc.transform(X)
    corr = np.corrcoef(Xc[0], ref[0])[0, 1]
    assert corr > 0.99
