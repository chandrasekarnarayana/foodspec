import importlib

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
