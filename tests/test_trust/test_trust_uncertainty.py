import numpy as np

from foodspec.trust.abstention import apply_abstention_rules
from foodspec.trust.calibration import PlattCalibrator
from foodspec.trust.conformal import MondrianConformalClassifier
from foodspec.trust.metrics import compute_calibration_metrics


def test_platt_calibration_reduces_ece():
    rng = np.random.default_rng(0)
    n = 4000
    p_true = rng.uniform(0.05, 0.95, size=n)
    y_true = rng.binomial(1, p_true)

    # Overconfident predictions (miscalibrated)
    p_pred = 1.0 / (1.0 + np.exp(-3.0 * (p_true - 0.5)))
    proba = np.column_stack([1.0 - p_pred, p_pred])

    idx = rng.permutation(n)
    split = int(0.4 * n)
    cal_idx = idx[:split]
    test_idx = idx[split:]

    calibrator = PlattCalibrator()
    calibrator.fit(y_true[cal_idx], proba[cal_idx])
    proba_cal = calibrator.transform(proba[test_idx])

    before = compute_calibration_metrics(y_true[test_idx], proba[test_idx])
    after = compute_calibration_metrics(y_true[test_idx], proba_cal)

    assert after["ece"] <= before["ece"]


def test_conformal_coverage_close_to_target():
    rng = np.random.default_rng(0)
    n_cal = 1000
    n_test = 1000
    n_classes = 3

    proba_cal = rng.dirichlet(np.ones(n_classes), size=n_cal)
    y_cal = np.array([rng.choice(n_classes, p=p) for p in proba_cal])
    proba_test = rng.dirichlet(np.ones(n_classes), size=n_test)
    y_test = np.array([rng.choice(n_classes, p=p) for p in proba_test])

    cp = MondrianConformalClassifier(alpha=0.1)
    cp.fit(y_cal, proba_cal)
    result = cp.predict_sets(proba_test, y_true=y_test)

    assert result.coverage is not None
    assert result.coverage >= 0.85


def test_abstention_risk_coverage_monotonic():
    rng = np.random.default_rng(0)
    n = 300
    n_classes = 4
    proba = rng.dirichlet(np.ones(n_classes), size=n)
    y_true = np.array([rng.choice(n_classes, p=p) for p in proba])

    result = apply_abstention_rules(proba, y_true, tau=0.6)
    coverage = result.risk_coverage["coverage"]

    assert all(coverage[i] >= coverage[i + 1] - 1e-12 for i in range(len(coverage) - 1))
