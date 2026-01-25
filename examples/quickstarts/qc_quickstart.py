"""
Quickstart script for QC/novelty detection using foodspec.
Run with: python examples/qc_quickstart.py
"""

import numpy as np
import pandas as pd

from foodspec.apps.qc import run_qc_workflow
from foodspec.core.dataset import FoodSpectrumSet


def _synthetic_qc():
    rng = np.random.default_rng(0)
    wavenumbers = np.linspace(600, 1800, 80)
    n_ref = 12
    n_test = 6
    ref_spectra = []
    for _ in range(n_ref):
        ref_spectra.append(
            np.exp(-0.5 * ((wavenumbers - 1200) / 20) ** 2) + rng.normal(0, 0.01, size=wavenumbers.shape)
        )
    test_spectra = []
    for i in range(n_test):
        if i < 3:
            test_spectra.append(
                np.exp(-0.5 * ((wavenumbers - 1200) / 20) ** 2) + rng.normal(0, 0.02, size=wavenumbers.shape)
            )
        else:
            # mild shift to emulate suspect
            test_spectra.append(
                np.exp(-0.5 * ((wavenumbers - 1250) / 18) ** 2) + rng.normal(0, 0.02, size=wavenumbers.shape)
            )
    x = np.vstack(ref_spectra + test_spectra)
    labels = ["auth_ref"] * n_ref + ["eval"] * n_test
    metadata = pd.DataFrame({"sample_id": [f"s{i:02d}" for i in range(len(x))], "group": labels})
    return FoodSpectrumSet(x=x, wavenumbers=wavenumbers, metadata=metadata, modality="raman")


def main():
    fs = _synthetic_qc()
    train_mask = fs.metadata["group"] == "auth_ref"
    res = run_qc_workflow(fs, train_mask=train_mask, model_type="oneclass_svm")
    print(res.labels_pred.value_counts())
    print("Threshold:", res.threshold)


if __name__ == "__main__":
    main()
