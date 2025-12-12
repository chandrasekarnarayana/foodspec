import numpy as np
import pandas as pd

from foodspec.harmonization import (
    CalibrationCurve,
    apply_calibration,
    harmonize_datasets_advanced,
    intensity_normalize_by_power,
)
from foodspec.spectral_dataset import SpectralDataset


def test_calibration_and_power_norm():
    wn = np.array([1000.0, 1010.0, 1020.0])
    spectra = np.array([[1.0, 2.0, 3.0]])
    meta = pd.DataFrame({"sample": [1]})
    ds = SpectralDataset(wn, spectra, meta, instrument_meta={"instrument_id": "inst1", "laser_power_mw": 10.0})
    curve = CalibrationCurve(instrument_id="inst1", wn_source=wn, wn_target=wn + 1.0)
    ds_cal = apply_calibration(ds, curve)
    assert np.allclose(ds_cal.wavenumbers, wn + 1.0)
    ds_pow = intensity_normalize_by_power(ds_cal, 10.0)
    assert np.allclose(ds_pow.spectra, spectra / 10.0)


def test_harmonize_advanced_residual_diag():
    wn = np.array([1000.0, 1010.0, 1020.0])
    spec1 = SpectralDataset(
        wn,
        np.array([[1.0, 2.0, 3.0]]),
        pd.DataFrame({"sample": [1]}),
        instrument_meta={"instrument_id": "A"},
    )
    spec2 = SpectralDataset(
        wn + 1.0,
        np.array([[1.1, 2.1, 3.1]]),
        pd.DataFrame({"sample": [2]}),
        instrument_meta={"instrument_id": "B"},
    )
    calib = {"B": CalibrationCurve(instrument_id="B", wn_source=wn + 1.0, wn_target=wn)}
    harmonized, diag = harmonize_datasets_advanced([spec1, spec2], calibration_curves=calib)
    assert len(harmonized) == 2
    # Wavenumbers should match target grid
    assert np.allclose(harmonized[0].wavenumbers, harmonized[1].wavenumbers)
    assert "residual_std_mean" in diag
