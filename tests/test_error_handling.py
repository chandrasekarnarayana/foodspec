"""Tests for error handling, hints, and prediction QC."""

import numpy as np

from foodspec.predict.guards import guard_prediction
from foodspec.qc import evaluate_prediction_qc
from foodspec.utils.errors import FriendlyError, friendly_error
from foodspec.utils.hints import suggest_fixes


def test_friendly_error_file_not_found():
    err = friendly_error(FileNotFoundError("missing.csv"), context="loading data")
    assert isinstance(err, FriendlyError)
    assert err.code == "file_not_found"
    assert "missing.csv" in err.message
    assert "Hint" in err.as_markdown()


def test_suggest_fixes():
    hint = suggest_fixes("import_error", context="starting CLI")
    assert "install" in hint.lower()
    assert "CLI" in hint


def test_guard_prediction_low_confidence():
    probs = np.array([0.4, 0.6])
    do_not_trust, warnings = guard_prediction(probs, min_confidence=0.7)
    assert do_not_trust
    assert any("Low maximum confidence" in w.reason for w in warnings)


def test_evaluate_prediction_qc_flags_drift():
    probs = np.array([0.3, 0.7])
    result = evaluate_prediction_qc(probs, drift_score=0.3, drift_threshold=0.2)
    assert result.do_not_trust
    assert any("drift" in r.lower() for r in result.reasons)


def test_evaluate_prediction_qc_calibration():
    probs = np.array([0.45, 0.55])
    result = evaluate_prediction_qc(probs, ece=0.1, ece_threshold=0.08)
    assert result.do_not_trust
    assert any("calibration" in r.lower() for r in result.reasons)
