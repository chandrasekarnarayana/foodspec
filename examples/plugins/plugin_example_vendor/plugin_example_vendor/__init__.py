"""
Example vendor loader plugin.
Adds a dummy loader for *.toy files (ASCII with two columns: meta, intensity).
"""
import pandas as pd
import numpy as np

from foodspec.spectral_dataset import SpectralDataset


def load_toy(path):
    df = pd.read_csv(path, sep=",")
    wn = np.array([1000.0, 1010.0])
    spectra = df[["int1", "int2"]].to_numpy()
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
