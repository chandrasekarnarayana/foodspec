"""
Notebook-style demo: load spectra, preprocess, extract peaks, run RQ engine.
"""

from foodspec.core.spectral_dataset import PreprocessOptions
from foodspec.features.rq import PeakDefinition, RatioDefinition, RatioQualityEngine, RQConfig
from foodspec.spectral_io import load_any_spectra


def main():
    ds = load_any_spectra("examples/demo_spectra.csv")  # replace with your path
    pp = PreprocessOptions(
        baseline_method="als",
        baseline_lambda=1e5,
        baseline_p=0.01,
        smoothing_method="savgol",
        smoothing_window=9,
        smoothing_polyorder=3,
        normalization="reference",
        reference_wavenumber=2720.0,
    )
    ds_proc = ds.preprocess(pp)

    peaks = [
        PeakDefinition(name="I_1742", column="I_1742", wavenumber=1742),
        PeakDefinition(name="I_1652", column="I_1652", wavenumber=1652),
        PeakDefinition(name="I_2720", column="I_2720", wavenumber=2720),
    ]
    peak_df = ds_proc.to_peaks(peaks)
    ratios = [
        RatioDefinition(name="1742/2720", numerator="I_1742", denominator="I_2720"),
        RatioDefinition(name="1652/2720", numerator="I_1652", denominator="I_2720"),
    ]
    cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage")
    res = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg).run_all(peak_df)
    print(res.text_report.splitlines()[:20])


if __name__ == "__main__":
    main()
