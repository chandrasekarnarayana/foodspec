import importlib
import sys

import pytest


def test_conv1d_requires_tensorflow(monkeypatch):
    # Simulate missing tensorflow
    def fake_find_spec(name):
        if name == "tensorflow":
            return None
        return importlib.machinery.ModuleSpec(name, None)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    from foodspec.chemometrics.deep import Conv1DSpectrumClassifier

    with pytest.raises(ImportError) as exc:
        Conv1DSpectrumClassifier()
    assert "requires TensorFlow" in str(exc.value)
    assert "pip install 'foodspec[deep]'" in str(exc.value)


def test_conv1d_with_fake_tensorflow(monkeypatch):
    import types
    import numpy as np

    class FakeSequential:
        def __init__(self, layers=None):
            self.layers = layers or []

        def compile(self, optimizer=None, loss=None, metrics=None):
            return None

        def fit(self, X, y, epochs=1, batch_size=1, validation_split=0.0, verbose=0):
            self.X_shape = X.shape
            self.y_shape = y.shape
            return None

        def predict(self, X, verbose=0):
            # simple deterministic softmax-like output
            n = X.shape[0]
            return np.tile(np.array([[0.3, 0.7]]), (n, 1))

    fake_layers = types.SimpleNamespace(
        Input=lambda shape=None: None,
        Conv1D=lambda *args, **kwargs: None,
        MaxPool1D=lambda *args, **kwargs: None,
        Flatten=lambda: None,
        Dense=lambda *args, **kwargs: None,
    )

    fake_utils = types.SimpleNamespace(set_random_seed=lambda seed: None)

    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            layers=fake_layers,
            models=types.SimpleNamespace(Sequential=FakeSequential),
            utils=fake_utils,
            Sequential=FakeSequential,
        )
    )

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)

    from foodspec.chemometrics.deep import Conv1DSpectrumClassifier

    X = np.random.rand(4, 10)
    y = np.array([0, 1, 0, 1])

    clf = Conv1DSpectrumClassifier(epochs=1, batch_size=1, validation_split=0.0)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    assert probs.shape == (4, 2)
    assert preds.shape == (4,)


def test_conv1d_predict_before_fit(monkeypatch):
    import types
    import numpy as np

    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            layers=types.SimpleNamespace(),
            models=types.SimpleNamespace(Sequential=lambda layers=None: None),
            utils=types.SimpleNamespace(set_random_seed=lambda seed: None),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)
    from foodspec.chemometrics.deep import Conv1DSpectrumClassifier

    clf = Conv1DSpectrumClassifier(epochs=1, batch_size=1, validation_split=0.0)
    with pytest.raises(RuntimeError):
        clf.predict(np.random.rand(2, 5))
