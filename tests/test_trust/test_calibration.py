import numpy as np

from foodspec.trust.calibration import TemperatureScaler


def test_temperature_scaler_fit_transform():
    logits = np.array([[2.0, 0.5], [0.1, 1.2]])
    labels = np.array([0, 1])
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    calibrated = scaler.transform(logits)
    assert calibrated.shape == logits.shape

