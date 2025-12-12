import numpy as np
import pandas as pd

from foodspec.preprocessing_pipeline import PreprocessingConfig, run_full_preprocessing, detect_input_mode


def test_preprocess_raw_spectra_produces_peaks():
    # synthetic wide-format spectra: metadata + wn
    df = pd.DataFrame(
        {
            "oil_type": ["A", "B"],
            "1000": [1.0, 2.0],
            "1742": [5.0, 6.0],
            "2720": [10.0, 10.0],
        }
    )
    cfg = PreprocessingConfig(
        baseline_enabled=False,
        smooth_enabled=False,
        normalization="reference",
        reference_wavenumber=2720.0,
        peak_definitions=None,
    )
    mode = detect_input_mode(df)
    assert mode == "raw_spectra"
    cfg.peak_definitions = []  # no peaks, should return processed spectra
    out = run_full_preprocessing(df, cfg)
    assert "oil_type" in out.columns
    assert "1742" in out.columns


def test_preprocess_with_peaks_extracts_intensity():
    df = pd.DataFrame(
        {
            "oil_type": ["A"],
            "1000": [1.0],
            "1742": [5.0],
            "1745": [6.0],
            "2720": [10.0],
        }
    )
    from foodspec.rq import PeakDefinition
    cfg = PreprocessingConfig(
        baseline_enabled=False,
        smooth_enabled=False,
        normalization="none",
        peak_definitions=[PeakDefinition(name="I_1742", column="I_1742", wavenumber=1742.0, window=(1738, 1746))],
    )
    out = run_full_preprocessing(df, cfg)
    assert "I_1742" in out.columns
    assert np.isclose(out["I_1742"].iloc[0], 6.0)
