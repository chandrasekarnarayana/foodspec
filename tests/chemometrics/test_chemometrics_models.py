import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from foodspec.chemometrics.models import _PLSProjector, make_classifier, make_pls_da


def test_classifier_factory_core_models():
    clf = make_classifier("logreg")
    assert isinstance(clf, LogisticRegression)

    clf = make_classifier("svm_linear")
    assert isinstance(clf, SVC) and clf.kernel == "linear"

    clf = make_classifier("svm_rbf")
    assert isinstance(clf, SVC) and clf.kernel == "rbf"

    clf = make_classifier("rf", n_estimators=10)
    assert isinstance(clf, RandomForestClassifier)

    clf = make_classifier("knn", n_neighbors=3)
    assert isinstance(clf, KNeighborsClassifier)


def test_classifier_factory_optional_models():
    for name in ("xgb", "lgbm"):
        try:
            clf = make_classifier(name)
        except ImportError:
            continue
        else:
            assert clf is not None


def test_pls_projector_vip_scores():
    X = np.random.RandomState(0).randn(20, 10)
    y = np.random.RandomState(0).randint(0, 2, 20)
    proj = _PLSProjector(n_components=5)
    proj.fit(X, y)
    vip = proj.get_vip_scores()
    assert vip.shape == (10,)
    assert np.all(vip >= 0), "VIP scores must be non-negative"
    assert np.sum(vip) > 0, "VIP sum must be positive"


def test_pls_da_integration_with_vip():
    X = np.random.RandomState(42).randn(30, 15)
    y = np.random.RandomState(42).randint(0, 3, 30)
    pipeline = make_pls_da(n_components=4)
    pipeline.fit(X, y)
    pred = pipeline.predict(X)
    assert len(pred) == len(X)
    pls_proj = pipeline.named_steps["pls_proj"]
    vip = pls_proj.get_vip_scores()
    assert vip.shape == (15,)
