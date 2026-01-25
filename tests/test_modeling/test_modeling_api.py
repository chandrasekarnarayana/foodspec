from foodspec.modeling import make_classifier


def test_make_classifier():
    model = make_classifier("logreg")
    assert model is not None

