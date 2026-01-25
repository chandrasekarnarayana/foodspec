import json
from pathlib import Path

import numpy as np

from foodspec.features.selection import StabilitySelector, run_stability_selection_cv


class SimpleEstimator:
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.coef_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        corrs = np.abs(np.corrcoef(X.T, y)[-1, :-1])
        thresh = np.quantile(corrs, 0.9)
        coef = np.zeros(X.shape[1])
        mask = corrs >= thresh
        coef[mask] = rng.uniform(0.1, 0.5, size=mask.sum())
        self.coef_ = coef
        return self


def test_cv_saves_marker_panel_with_names(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 40))
    y = rng.integers(0, 2, size=100)
    names = [f"f{i}" for i in range(X.shape[1])]

    selector = StabilitySelector(
        estimator_factory=lambda: SimpleEstimator(random_state=7),
        n_resamples=8,
        subsample_fraction=0.7,
        selection_threshold=0.3,
        random_state=99,
    )

    outdir = tmp_path / "cv_run"
    panel = run_stability_selection_cv(
        estimator_factory=lambda **p: SimpleEstimator(random_state=7),
        selector=selector,
        X=X,
        y=y,
        x_wavenumbers=np.linspace(1000, 2000, X.shape[1]),
        n_splits=3,
        seed=123,
        output_dir=outdir,
        feature_names=names,
    )

    # Legacy return still has indices and frequencies
    assert "selected_indices" in panel and "selection_frequencies" in panel

    # Saved artifact exists and contains selected_feature_names field
    mp_path = outdir / "marker_panel.json"
    assert mp_path.exists()
    data = json.loads(mp_path.read_text())
    assert "selected_feature_names" in data
    # Frequencies are present and in [0,1]
    freqs = np.array(data["selection_frequencies"]) 
    assert freqs.min() >= 0.0 and freqs.max() <= 1.0
