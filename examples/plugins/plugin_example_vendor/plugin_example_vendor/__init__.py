"""
Example vendor loader plugin.
Adds a dummy loader for *.toy files (CSV with columns: meta, int1, int2, ...).
"""
import pandas as pd
import numpy as np

from foodspec.spectral_dataset import SpectralDataset


def load_toy(path):
    df = pd.read_csv(path, sep=",")
    # Use all columns prefixed with 'int' as spectral intensities
    int_cols = [c for c in df.columns if c.startswith("int")]
    if len(int_cols) < 3:
        # Ensure at least 3 points to satisfy validation downstream
        raise ValueError("Toy plugin expects at least three intensity columns: int1,int2,int3")
    n_points = len(int_cols)
    wn = np.linspace(1000.0, 1000.0 + 10.0 * (n_points - 1), n_points)
    spectra = df[int_cols].to_numpy()
    meta = df[["meta"]]
    return SpectralDataset(wn, spectra, meta)


def get_plugins():
    return {
        "protocols": [],
        "vendor_loaders": {"toy": load_toy},
        "harmonization": {},
    }


def plugin_main():
    return get_plugins()
