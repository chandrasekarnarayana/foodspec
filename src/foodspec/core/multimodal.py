"""Multi-modal spectroscopy dataset support.

Enables workflows combining Raman, FTIR, and NIR on the same samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet, Modality


@dataclass
class MultiModalDataset:
    """Spectral datasets from multiple modalities with aligned metadata.

    Each modality (Raman, FTIR, NIR) is stored as a separate `FoodSpectrumSet`.
    Metadata is aligned across modalities via a common sample identifier.

    Args:
        datasets (dict[Modality, FoodSpectrumSet]): Mapping modality → dataset.
        metadata (pd.DataFrame): One row per sample aligned across modalities.
        sample_id_col (str): Column used to align samples across datasets.
    """

    datasets: Dict[Modality, FoodSpectrumSet]
    metadata: pd.DataFrame
    sample_id_col: str = "sample_id"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.datasets:
            raise ValueError("datasets must contain at least one modality.")
        if self.sample_id_col not in self.metadata.columns:
            raise ValueError(f"sample_id_col '{self.sample_id_col}' not found in metadata.")
        n_samples = len(self.metadata)
        for modality, ds in self.datasets.items():
            if len(ds) != n_samples:
                raise ValueError(f"Dataset for {modality} has {len(ds)} samples, but metadata has {n_samples} rows.")

    def modalities(self) -> List[Modality]:
        """Return list of available modalities.

        Returns:
            list[Modality]: Modalities present.
        """
        return list(self.datasets.keys())

    def get_modality(self, modality: Modality) -> FoodSpectrumSet:
        """Retrieve dataset for a specific modality.

        Args:
            modality (Modality): Modality key.

        Returns:
            FoodSpectrumSet: Dataset for requested modality.

        Raises:
            ValueError: If modality not found.
        """
        if modality not in self.datasets:
            raise ValueError(f"Modality '{modality}' not found in dataset.")
        return self.datasets[modality]

    def subset_samples(self, indices: Sequence[int]) -> "MultiModalDataset":
        """Return a subset of samples by index.

        Args:
            indices (Sequence[int]): Zero-based sample indices.

        Returns:
            MultiModalDataset: New dataset with subsets applied to all modalities.

        Raises:
            ValueError: If indices out of range.
        """
        indices_arr = np.array(indices, dtype=int)
        if np.any(indices_arr < 0) or np.any(indices_arr >= len(self.metadata)):
            raise ValueError("indices contain out-of-range values.")
        meta_sub = self.metadata.iloc[indices_arr].reset_index(drop=True)
        datasets_sub = {modality: ds.subset(indices=indices_arr.tolist()) for modality, ds in self.datasets.items()}
        return MultiModalDataset(
            datasets=datasets_sub,
            metadata=meta_sub,
            sample_id_col=self.sample_id_col,
        )

    def filter_by_metadata(self, **filters) -> "MultiModalDataset":
        """Filter samples by metadata column values.

        Keyword Args:
            **filters: Column → value constraints; sequences use membership.

        Returns:
            MultiModalDataset: Filtered dataset.

        Raises:
            ValueError: If a column not found in metadata.
        """
        mask = np.ones(len(self.metadata), dtype=bool)
        for col, value in filters.items():
            if col not in self.metadata.columns:
                raise ValueError(f"Column '{col}' not found in metadata.")
            if isinstance(value, (list, tuple, set)):
                mask &= self.metadata[col].isin(value).to_numpy()
            else:
                mask &= (self.metadata[col] == value).to_numpy()
        indices = np.where(mask)[0]
        return self.subset_samples(indices)

    @classmethod
    def from_datasets(
        cls,
        datasets: Dict[Modality, FoodSpectrumSet],
        sample_id_col: str = "sample_id",
    ) -> "MultiModalDataset":
        """Construct from dict of datasets, aligning by sample identifier.

        Args:
            datasets (dict[Modality, FoodSpectrumSet]): Input datasets.
            sample_id_col (str): Column with sample IDs.

        Returns:
            MultiModalDataset: Aligned multi-modal dataset.

        Raises:
            ValueError: If inputs missing, sample_id_col missing, or IDs misaligned.
        """
        if not datasets:
            raise ValueError("datasets must contain at least one modality.")
        # Use first dataset's metadata as template
        ref_modality = list(datasets.keys())[0]
        ref_ds = datasets[ref_modality]
        if sample_id_col not in ref_ds.metadata.columns:
            raise ValueError(f"sample_id_col '{sample_id_col}' not in reference dataset metadata.")
        metadata = ref_ds.metadata.copy()
        # Validate all datasets have same sample IDs in same order
        ref_ids = metadata[sample_id_col].tolist()
        for modality, ds in datasets.items():
            if sample_id_col not in ds.metadata.columns:
                raise ValueError(f"sample_id_col '{sample_id_col}' not in {modality} dataset metadata.")
            ids = ds.metadata[sample_id_col].tolist()
            if ids != ref_ids:
                raise ValueError(
                    f"Sample IDs for {modality} do not match reference. "
                    "Ensure all datasets have identical sample order."
                )
        return cls(datasets=datasets, metadata=metadata, sample_id_col=sample_id_col)

    def to_feature_dict(self) -> Dict[Modality, np.ndarray]:
        """Return spectral matrices as dict for fusion.

        Returns:
            dict[Modality, np.ndarray]: Modality → feature matrix.
        """
        return {modality: ds.x for modality, ds in self.datasets.items()}


__all__ = ["MultiModalDataset"]
