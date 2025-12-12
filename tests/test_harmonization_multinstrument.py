import numpy as np
import pandas as pd

from foodspec.harmonization import CalibrationCurve, harmonize_datasets_advanced
from foodspec.spectral_dataset import SpectralDataset


def test_cross_instrument_harmonization_reduces_rmse(tmp_path):
    wn_ref = np.array([1000.0, 1010.0, 1020.0])
    true_spec = np.array([1.0, 2.0, 3.0])

    # Instrument A: no shift, power 1
    ds_a = SpectralDataset(wn_ref, true_spec[None, :], pd.DataFrame({"sample": [1]}), instrument_meta={"instrument_id": "A", "laser_power_mw": 1.0})
    # Instrument B: shifted by +2 cm^-1 and scaled by 2x power
    ds_b = SpectralDataset(wn_ref + 2.0, (true_spec * 2)[None, :], pd.DataFrame({"sample": [2]}), instrument_meta={"instrument_id": "B", "laser_power_mw": 2.0})

    calib = {
        "B": CalibrationCurve(instrument_id="B", wn_source=wn_ref + 2.0, wn_target=wn_ref)
    }

    # Before harmonization RMSE
    rmse_before = np.sqrt(np.mean((ds_b.spectra - ds_a.spectra) ** 2))

    harmonized, diag = harmonize_datasets_advanced([ds_a, ds_b], calibration_curves=calib)
    rmse_after = np.sqrt(np.mean((harmonized[1].spectra - harmonized[0].spectra) ** 2))
    assert rmse_after < rmse_before
    # Harmonization params should be recorded
    assert "harmonization_params" in harmonized[1].instrument_meta
    assert "residual_std_mean" in diag
