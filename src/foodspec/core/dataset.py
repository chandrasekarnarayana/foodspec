"""Core data model for spectral datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

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
    metadata: Optional[pd.DataFrame] = None
    modality: Modality = "raman"
    label_col: Optional[str] = None
    group_col: Optional[str] = None
    batch_col: Optional[str] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = pd.DataFrame(index=np.arange(self.x.shape[0]))
        else:
            # Preserve reference semantics for shallow copies by resetting index in place
            # This avoids creating a new DataFrame that would break aliasing expectations.
            self.metadata.reset_index(drop=True, inplace=True)

        # Auto-detect common label/group columns if not specified
        if self.label_col is None and hasattr(self.metadata, "columns"):
            for candidate in ["label", "class", "target"]:
                if candidate in self.metadata.columns:
                    self.label_col = candidate
                    break
        if self.group_col is None and hasattr(self.metadata, "columns"):
            for candidate in ["group", "split", "fold"]:
                if candidate in self.metadata.columns:
                    self.group_col = candidate
                    break
        if self.batch_col is None and hasattr(self.metadata, "columns"):
            for candidate in ["batch_id", "batch", "run_id"]:
                if candidate in self.metadata.columns:
                    self.batch_col = candidate
                    break

        self.validate()

    def __len__(self) -> int:
        """Number of spectra in the set."""

        return self.x.shape[0]

    @property
    def labels(self) -> Optional[pd.Series]:
        """Return label column if configured."""

        if self.label_col and self.label_col in self.metadata.columns:
            return self.metadata[self.label_col]
        return None

    @property
    def groups(self) -> Optional[pd.Series]:
        """Return grouping column if configured."""

        if self.group_col and self.group_col in self.metadata.columns:
            return self.metadata[self.group_col]
        return None

    @property
    def batch_ids(self) -> Optional[pd.Series]:
        """Return batch identifier column if configured."""

        if self.batch_col and self.batch_col in self.metadata.columns:
            return self.metadata[self.batch_col]
        return None

    def _copy_with(
        self,
        *,
        x: Optional[np.ndarray] = None,
        wavenumbers: Optional[np.ndarray] = None,
        metadata: Optional[pd.DataFrame] = None,
    ) -> "FoodSpectrumSet":
        return FoodSpectrumSet(
            x=x if x is not None else self.x,
            wavenumbers=wavenumbers if wavenumbers is not None else self.wavenumbers,
            metadata=metadata if metadata is not None else self.metadata.copy(deep=True),
            modality=self.modality,
            label_col=self.label_col,
            group_col=self.group_col,
            batch_col=self.batch_col,
        )

    def __getitem__(self, index: IndexType) -> "FoodSpectrumSet":
        """Return a subset by position index or slice."""

        indices = self._normalize_index(index)
        return self._copy_with(
            x=self.x[indices],
            wavenumbers=self.wavenumbers.copy(),
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
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
                indices_array = np.array([idx for idx in indices_array if mask[idx]], dtype=int)
            selected_indices = indices_array
        else:
            selected_indices = np.where(mask)[0]

        return self._copy_with(
            x=self.x[selected_indices],
            wavenumbers=self.wavenumbers.copy(),
            metadata=self.metadata.iloc[selected_indices].reset_index(drop=True),
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
            x=x,
            wavenumbers=wavenumbers,
            metadata=metadata,
            modality=self.modality,
            label_col=self.label_col,
            group_col=self.group_col,
            batch_col=self.batch_col,
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
        return pd.concat([self.metadata.reset_index(drop=True).copy(), spectra_df], axis=1)

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
        if np.any(np.diff(self.wavenumbers) <= 0):
            raise ValueError("wavenumbers must be strictly increasing (monotonic).")
        if len(self.metadata) != n_samples:
            raise ValueError(
                f"metadata length does not match number of rows in x ({len(self.metadata)} != {n_samples})."
            )
        if self.modality not in {"raman", "ftir", "nir"}:
            raise ValueError(f"modality must be one of {{'raman', 'ftir', 'nir'}}; got '{self.modality}'.")
        for col_name in [self.label_col, self.group_col, self.batch_col]:
            if col_name is not None and col_name not in self.metadata.columns:
                raise ValueError(f"metadata column '{col_name}' not found for configured annotations.")

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

    def apply(self, func: Callable[[np.ndarray], np.ndarray], *, inplace: bool = False) -> "FoodSpectrumSet":
        """Apply a vectorized operation to all spectra."""

        result = np.asarray(func(self.x))
        if result.shape != self.x.shape:
            raise ValueError("apply must return an array with the same shape as x.")
        if inplace:
            self.x = result
            return self
        return self._copy_with(x=result)

    def scale(self, factor: float, *, inplace: bool = False) -> "FoodSpectrumSet":
        """Scale spectral intensities by a factor."""

        return self.apply(lambda arr: arr * factor, inplace=inplace)

    def offset(self, value: float, *, inplace: bool = False) -> "FoodSpectrumSet":
        """Add a constant offset to spectral intensities."""

        return self.apply(lambda arr: arr + value, inplace=inplace)

    def add_metadata_column(self, name: str, values: Sequence[Any], *, overwrite: bool = False) -> "FoodSpectrumSet":
        """Attach a metadata column aligned with spectra."""

        if name in self.metadata.columns and not overwrite:
            raise ValueError(f"metadata column '{name}' already exists; set overwrite=True to replace.")
        if len(values) != len(self):
            raise ValueError("metadata column length must match number of samples.")
        meta = self.metadata.copy(deep=True)
        meta[name] = list(values)
        return self._copy_with(metadata=meta)

    def select_wavenumber_range(self, min_wn: float, max_wn: float) -> "FoodSpectrumSet":
        """Return spectra restricted to a wavenumber window."""

        if min_wn > max_wn:
            raise ValueError("min_wn must be <= max_wn.")
        mask = (self.wavenumbers >= min_wn) & (self.wavenumbers <= max_wn)
        if not np.any(mask):
            raise ValueError("No wavenumbers fall within the requested range.")
        return self._copy_with(x=self.x[:, mask], wavenumbers=self.wavenumbers[mask])

    def with_annotations(
        self,
        *,
        label_col: Optional[str] = None,
        group_col: Optional[str] = None,
        batch_col: Optional[str] = None,
    ) -> "FoodSpectrumSet":
        """Return a copy with updated label/group/batch annotations."""

        return FoodSpectrumSet(
            x=self.x,
            wavenumbers=self.wavenumbers,
            metadata=self.metadata.copy(deep=True),
            modality=self.modality,
            label_col=label_col or self.label_col,
            group_col=group_col or self.group_col,
            batch_col=batch_col or self.batch_col,
        )

    @classmethod
    def concat(cls, datasets: Sequence["FoodSpectrumSet"]) -> "FoodSpectrumSet":
        """Concatenate multiple datasets with shared wavenumber grids."""

        if not datasets:
            raise ValueError("datasets must be non-empty.")
        ref = datasets[0]
        for ds in datasets[1:]:
            if ds.wavenumbers.shape != ref.wavenumbers.shape or not np.allclose(ds.wavenumbers, ref.wavenumbers):
                raise ValueError("All datasets must share identical wavenumber grids to concatenate.")
        x = np.vstack([ds.x for ds in datasets])
        metadata = pd.concat([ds.metadata for ds in datasets], ignore_index=True)
        return cls(
            x=x,
            wavenumbers=ref.wavenumbers.copy(),
            metadata=metadata.reset_index(drop=True),
            modality=ref.modality,
            label_col=ref.label_col,
            group_col=ref.group_col,
            batch_col=ref.batch_col,
        )

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
            label_col=self.label_col,
            group_col=self.group_col,
            batch_col=self.batch_col,
        )
        return train_ds, test_ds

    def to_hdf5(self, path: Union[str, Path], key: str = "foodspec", mode: str = "w", complevel: int = 4) -> Path:
        """Persist dataset to HDF5 (lazy-friendly storage)."""

        path = Path(path)
        df_x = pd.DataFrame(self.x)
        df_wn = pd.DataFrame({"wavenumber": self.wavenumbers})
        df_meta = self.metadata.reset_index(drop=True).copy()
        df_info = pd.DataFrame(
            [
                {
                    "modality": self.modality,
                    "label_col": self.label_col or "",
                    "group_col": self.group_col or "",
                    "batch_col": self.batch_col or "",
                }
            ]
        )
        with pd.HDFStore(path, mode=mode, complevel=complevel, complib="zlib") as store:
            store.put(f"{key}_x", df_x, format="table")
            store.put(f"{key}_wn", df_wn, format="table")
            store.put(f"{key}_meta", df_meta, format="table")
            store.put(f"{key}_info", df_info, format="table")
        return path

    @classmethod
    def from_hdf5(cls, path: Union[str, Path], key: str = "foodspec") -> "FoodSpectrumSet":
        """Load dataset from HDF5 created by ``to_hdf5``."""

        path = Path(path)
        with pd.HDFStore(path, mode="r") as store:
            df_x = store.get(f"{key}_x")
            df_wn = store.get(f"{key}_wn")
            df_meta = store.get(f"{key}_meta")
            df_info = store.get(f"{key}_info") if f"{key}_info" in store else pd.DataFrame()
        info = df_info.iloc[0].to_dict() if not df_info.empty else {}
        modality = info.get("modality", "raman")
        label_col = info.get("label_col") or None
        group_col = info.get("group_col") or None
        batch_col = info.get("batch_col") or None
        return cls(
            x=df_x.to_numpy(),
            wavenumbers=df_wn["wavenumber"].to_numpy(),
            metadata=df_meta.reset_index(drop=True),
            modality=modality,
            label_col=label_col,
            group_col=group_col,
            batch_col=batch_col,
        )

    def to_parquet(self, path: Union[str, Path]) -> Path:
        """Persist dataset to Parquet using wide layout."""

        path = Path(path)
        wide = self.to_wide_dataframe().copy()
        wide["__fs_modality"] = self.modality
        wide["__fs_label_col"] = self.label_col or ""
        wide["__fs_group_col"] = self.group_col or ""
        wide["__fs_batch_col"] = self.batch_col or ""
        wide.to_parquet(path)
        return path

    @classmethod
    def from_parquet(cls, path: Union[str, Path]) -> "FoodSpectrumSet":
        """Load dataset from Parquet created by ``to_parquet``."""

        path = Path(path)
        df = pd.read_parquet(path)
        info_cols = [c for c in df.columns if c.startswith("__fs_")]
        info = {col: df[col].iloc[0] for col in info_cols}
        df = df.drop(columns=info_cols)
        intensity_cols = [c for c in df.columns if str(c).startswith("int_")]
        wavenumbers = np.array([float(str(c).split("int_")[1]) for c in intensity_cols])
        spectra = df[intensity_cols].to_numpy()
        metadata = df.drop(columns=intensity_cols).reset_index(drop=True)
        return cls(
            x=spectra,
            wavenumbers=wavenumbers,
            metadata=metadata,
            modality=info.get("__fs_modality", "raman"),
            label_col=info.get("__fs_label_col") or None,
            group_col=info.get("__fs_group_col") or None,
            batch_col=info.get("__fs_batch_col") or None,
        )


def to_sklearn(
    ds: FoodSpectrumSet,
    label_col: Optional[str] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (X, y) arrays suitable for scikit-learn.

    Parameters
    ----------
    ds : FoodSpectrumSet
        Dataset to convert.
    label_col : Optional[str]
        Column to use for labels; if None, uses ds.label_col if available.

    Returns
    -------
    (X, y)
        X shape (n_samples, n_features). y is None if label column not found.
    """

    X = np.asarray(ds.x, dtype=float)
    col = label_col or ds.label_col
    y = None
    if col and col in ds.metadata.columns:
        y = ds.metadata[col].to_numpy()
    return X, y


def from_sklearn(
    X: np.ndarray,
    y: Optional[Sequence] = None,
    wavenumbers: Sequence[float] = (),
    modality: Modality = "raman",
    labels_name: str = "label",
) -> FoodSpectrumSet:
    """Create a FoodSpectrumSet from scikit-learn style inputs.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_wavenumbers).
    y : Optional[Sequence]
        Optional labels aligned to rows in X.
    wavenumbers : Sequence[float]
        Spectral axis values; if empty, uses 0..n_features-1.
    modality : Modality
        Modality tag (e.g., 'raman').
    labels_name : str
        Name of the label column in metadata if y is provided.
    """

    X = np.asarray(X, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    if wn.size == 0:
        wn = np.arange(X.shape[1], dtype=float)
    meta = pd.DataFrame(index=np.arange(X.shape[0]))
    if y is not None:
        meta[labels_name] = list(y)
    return FoodSpectrumSet(
        x=X, wavenumbers=wn, metadata=meta, modality=modality, label_col=(labels_name if y is not None else None)
    )


__all__ = ["FoodSpectrumSet", "to_sklearn", "from_sklearn"]
