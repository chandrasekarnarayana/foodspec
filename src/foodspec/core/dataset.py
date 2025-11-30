"""Core data model for spectral datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Modality = Literal["raman", "ftir", "nir"]
IndexType = Union[int, slice]


@dataclass
class FoodSpectrumSet:
    """Collection of spectra with aligned metadata and axis information.

    Parameters
    ----------
    x :
        Array of shape (n_samples, n_wavenumbers) containing spectral intensities.
    wavenumbers :
        Array of shape (n_wavenumbers,) with the spectral axis values.
    metadata :
        DataFrame with one row per sample storing labels and acquisition info.
    modality :
        Spectroscopy modality identifier: ``"raman"``, ``"ftir"``, or ``"nir"``.
    """

    x: np.ndarray
    wavenumbers: np.ndarray
    metadata: pd.DataFrame
    modality: Modality

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        """Number of spectra in the set."""

        return self.x.shape[0]

    def __getitem__(self, index: IndexType) -> "FoodSpectrumSet":
        """Return a subset by position index or slice."""

        indices = self._normalize_index(index)
        return FoodSpectrumSet(
            x=self.x[indices],
            wavenumbers=self.wavenumbers.copy(),
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
            modality=self.modality,
        )

    def subset(
        self,
        by: Optional[Dict[str, Any]] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> "FoodSpectrumSet":
        """Subset the dataset by metadata filters and/or explicit indices.

        Parameters
        ----------
        by :
            Mapping of metadata column names to desired values. For sequence-like
            values, membership (``isin``) is applied; otherwise equality is used.
        indices :
            Explicit indices to retain. If provided together with ``by``, the
            intersection is taken in the order of ``indices``.

        Returns
        -------
        FoodSpectrumSet
            A new dataset containing the selected spectra.
        """

        if by is None and indices is None:
            return self.copy(deep=False)

        mask = np.ones(len(self), dtype=bool)
        if by is not None:
            for key, value in by.items():
                if key not in self.metadata.columns:
                    raise ValueError(f"Metadata column '{key}' not found.")
                series = self.metadata[key]
                if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                    mask &= series.isin(value).to_numpy()
                else:
                    mask &= (series == value).to_numpy()

        if indices is not None:
            indices_array = np.asarray(indices, dtype=int)
            if indices_array.ndim != 1:
                raise ValueError("indices must be a 1D sequence of integers.")
            if np.any(indices_array < 0) or np.any(indices_array >= len(self)):
                raise ValueError("indices contain out-of-range values.")
            if by is not None:
                indices_array = np.array(
                    [idx for idx in indices_array if mask[idx]], dtype=int
                )
            selected_indices = indices_array
        else:
            selected_indices = np.where(mask)[0]

        return FoodSpectrumSet(
            x=self.x[selected_indices],
            wavenumbers=self.wavenumbers.copy(),
            metadata=self.metadata.iloc[selected_indices].reset_index(drop=True),
            modality=self.modality,
        )

    def copy(self, deep: bool = True) -> "FoodSpectrumSet":
        """Return a copy of the dataset.

        Parameters
        ----------
        deep :
            If True, copy underlying arrays and metadata; otherwise reuse references.

        Returns
        -------
        FoodSpectrumSet
            Copied dataset.
        """

        if deep:
            x = np.array(self.x, copy=True)
            wavenumbers = np.array(self.wavenumbers, copy=True)
            metadata = self.metadata.copy(deep=True)
        else:
            x = self.x
            wavenumbers = self.wavenumbers
            metadata = self.metadata

        return FoodSpectrumSet(
            x=x, wavenumbers=wavenumbers, metadata=metadata, modality=self.modality
        )

    def to_wide_dataframe(self) -> pd.DataFrame:
        """Convert the dataset to a wide DataFrame.

        Returns
        -------
        pandas.DataFrame
            Metadata columns followed by one column per wavenumber named
            ``int_<wavenumber>``.
        """

        intensity_columns = [f"int_{float(wn)}" for wn in self.wavenumbers]
        spectra_df = pd.DataFrame(self.x, columns=intensity_columns)
        return pd.concat(
            [self.metadata.reset_index(drop=True).copy(), spectra_df], axis=1
        )

    def validate(self) -> None:
        """Validate array shapes, metadata length, and modality."""

        if self.x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_wavenumbers).")
        if self.wavenumbers.ndim != 1:
            raise ValueError("wavenumbers must be a 1D array.")
        n_samples, n_wavenumbers = self.x.shape
        if n_wavenumbers < 3:
            raise ValueError("At least three wavenumber points are required.")
        if self.wavenumbers.shape[0] != n_wavenumbers:
            raise ValueError(
                "wavenumbers length does not match number of columns in x "
                f"({self.wavenumbers.shape[0]} != {n_wavenumbers})."
            )
        if len(self.metadata) != n_samples:
            raise ValueError(
                "metadata length does not match number of rows in x "
                f"({len(self.metadata)} != {n_samples})."
            )
        if self.modality not in {"raman", "ftir", "nir"}:
            raise ValueError(
                "modality must be one of {'raman', 'ftir', 'nir'}; "
                f"got '{self.modality}'."
            )

    def _normalize_index(self, index: IndexType) -> np.ndarray:
        """Normalize indexing input to an array of indices."""

        if isinstance(index, int):
            if index < 0 or index >= len(self):
                raise IndexError("index out of range.")
            return np.array([index])
        if isinstance(index, slice):
            return np.arange(len(self))[index]
        raise TypeError("Index must be an integer or slice.")

    def to_X_y(self, target_col: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) for a target column in metadata."""

        if target_col not in self.metadata.columns:
            raise ValueError(f"Target column '{target_col}' not found in metadata.")
        return self.x, self.metadata[target_col].to_numpy()

    def train_test_split(
        self,
        target_col: str,
        test_size: float = 0.3,
        stratify: bool = True,
        random_state: Optional[int] = None,
    ) -> tuple["FoodSpectrumSet", "FoodSpectrumSet"]:
        """Split into train/test FoodSpectrumSets."""

        X, y = self.to_X_y(target_col)
        stratify_arg = y if stratify else None
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X,
            y,
            self.metadata,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )
        train_ds = FoodSpectrumSet(
            x=X_train,
            wavenumbers=self.wavenumbers.copy(),
            metadata=meta_train.reset_index(drop=True),
            modality=self.modality,
        )
        test_ds = FoodSpectrumSet(
            x=X_test,
            wavenumbers=self.wavenumbers.copy(),
            metadata=meta_test.reset_index(drop=True),
            modality=self.modality,
        )
        return train_ds, test_ds


__all__ = ["FoodSpectrumSet"]
