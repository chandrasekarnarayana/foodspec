import numpy as np
import pandas as pd

from foodspec.preprocessing_pipeline import PreprocessingConfig, run_full_preprocessing


def test_spike_removal_toggle_affects_output_and_counts():
    # Construct a tiny wide-format dataframe with a clear spike
    wn = [1000, 1005, 1010, 1015, 1020]
    X = np.array([
        [1.0, 1.2, 50.0, 1.1, 1.0],  # row with spike at 1010
        [0.9, 1.1, 0.95, 1.0, 0.97],  # normal row
    ])
    meta = pd.DataFrame({"oil_type": ["A", "B"]})
    df = pd.concat([meta, pd.DataFrame(X, columns=[str(w) for w in wn])], axis=1)

    # Spike removal ON
    cfg_on = PreprocessingConfig(
        baseline_method="none",
        baseline_enabled=False,
        smooth_enabled=False,
        normalization="none",
        spike_removal=True,
        spike_zscore_thresh=5.0,
    )
    proc_on = run_full_preprocessing(df, cfg_on)
    assert "spikes_removed" in proc_on.columns
    # cleaned value should be substantially reduced vs original spike
    row0_values_on = proc_on[[str(w) for w in wn]].iloc[0].to_numpy(dtype=float)
    assert row0_values_on[2] < 10.0  # spike corrected

    # Spike removal OFF
    cfg_off = PreprocessingConfig(
        baseline_method="none",
        baseline_enabled=False,
        smooth_enabled=False,
        normalization="none",
        spike_removal=False,
    )
    proc_off = run_full_preprocessing(df, cfg_off)
    row0_values_off = proc_off[[str(w) for w in wn]].iloc[0].to_numpy(dtype=float)
    # original spike should remain
    assert row0_values_off[2] > 40.0
    # when disabled, either column absent or zero count; accept both
    if "spikes_removed" in proc_off.columns:
        assert int(proc_off["spikes_removed"].iloc[0]) == 0
