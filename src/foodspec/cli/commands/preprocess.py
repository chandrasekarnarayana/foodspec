"""Preprocessing command: load, preprocess, save to HDF5."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from sklearn.pipeline import Pipeline

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.io import load_folder
from foodspec.io.exporters import to_hdf5
from foodspec.logging_utils import get_logger
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.cropping import RangeCropper
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother

logger = get_logger(__name__)

preprocess_app = typer.Typer(help="Preprocessing commands")


class _CropperWrapper(RangeCropper):
    """Pipeline-friendly wrapper over RangeCropper that stores axis."""

    def __init__(self, wavenumbers: np.ndarray, min_wn: float, max_wn: float):
        self.wavenumbers_full = np.asarray(wavenumbers, dtype=float)
        super().__init__(min_wn=min_wn, max_wn=max_wn)
        mask = (self.wavenumbers_full >= self.min_wn) & (self.wavenumbers_full <= self.max_wn)
        if not np.any(mask):
            raise ValueError("Cropping mask is empty.")
        self.mask_ = mask
        self.wavenumbers_ = self.wavenumbers_full[mask]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.wavenumbers_full.shape[0]:
            raise ValueError("Input X columns must match length of original wavenumbers.")
        return X[:, self.mask_]


def _default_preprocess_pipeline(wavenumbers: np.ndarray, min_wn: float, max_wn: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("als", ALSBaseline(lambda_=1e5, p=0.01, max_iter=10)),
            ("savgol", SavitzkyGolaySmoother(window_length=9, polyorder=3)),
            ("norm", VectorNormalizer(norm="l2")),
            ("crop", _CropperWrapper(wavenumbers=wavenumbers, min_wn=min_wn, max_wn=max_wn)),
        ]
    )


@preprocess_app.command("preprocess")
def preprocess(
    input_folder: str = typer.Argument(..., help="Folder containing spectra text files."),
    metadata_csv: Optional[str] = typer.Option(None, help="Optional metadata CSV with sample_id."),
    output_hdf5: str = typer.Argument(..., help="Output HDF5 path."),
    modality: str = typer.Option("raman", help="Spectroscopy modality."),
    min_wn: float = typer.Option(600.0, help="Minimum wavenumber for cropping."),
    max_wn: float = typer.Option(1800.0, help="Maximum wavenumber for cropping."),
):
    """Load spectra, apply default preprocessing, and save to HDF5."""
    ds = load_folder(
        folder=input_folder,
        metadata_csv=metadata_csv,
        modality=modality,
    )
    pipe = _default_preprocess_pipeline(ds.wavenumbers, min_wn=min_wn, max_wn=max_wn)
    x_proc = pipe.fit_transform(ds.x)
    cropper = pipe.named_steps["crop"]
    ds_out = FoodSpectrumSet(
        x=x_proc,
        wavenumbers=cropper.wavenumbers_,
        metadata=ds.metadata.copy(),
        modality=ds.modality,
    )
    to_hdf5(ds_out, output_hdf5)
    typer.echo(f"Preprocessed spectra saved to {output_hdf5}")
