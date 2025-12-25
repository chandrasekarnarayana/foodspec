from sklearn.datasets import make_classification, make_regression

from foodspec.chemometrics.models import make_classifier, make_mlp_regressor


def test_mlp_classifier_factory_runs():
    X, y = make_classification(n_samples=60, n_features=8, n_informative=4, random_state=0)
    clf = make_classifier("mlp", hidden_layer_sizes=(20,), max_iter=300)
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert acc > 0.7


def test_mlp_regressor_runs():
    X, y = make_regression(n_samples=80, n_features=6, n_informative=4, noise=0.1, random_state=0)
    reg = make_mlp_regressor(hidden_layer_sizes=(30,), max_iter=300)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == y.shape
