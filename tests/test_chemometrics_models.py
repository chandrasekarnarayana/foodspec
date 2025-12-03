from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from foodspec.chemometrics.models import make_classifier


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
