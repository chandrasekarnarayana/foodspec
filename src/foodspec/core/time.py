from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet


@dataclass
class TimeSpectrumSet(FoodSpectrumSet):
    """FoodSpectrumSet with explicit time indexing support.

    Adds a ``time_col`` indicating the metadata column containing time values.
    Provides helpers to sort by time and to extract per-entity trajectories.
    """

    time_col: Optional[str] = None
    entity_col: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.time_col is None:
            for candidate in ["time", "t", "minutes", "hours", "days"]:
                if candidate in self.metadata.columns:
                    self.time_col = candidate
                    break
        if self.entity_col is None:
            for candidate in ["sample_id", "batch_id", "run_id"]:
                if candidate in self.metadata.columns:
                    self.entity_col = candidate
                    break
        self._validate_time()

    def _validate_time(self) -> None:
        if self.time_col is None or self.time_col not in self.metadata.columns:
            raise ValueError("time_col not set or missing in metadata.")
        if not np.issubdtype(self.metadata[self.time_col].dtype, np.number):
            raise ValueError("time column must be numeric.")

    def sort_by_time(self) -> "TimeSpectrumSet":
        idx = np.argsort(self.metadata[self.time_col].to_numpy())
        meta = self.metadata.iloc[idx].reset_index(drop=True)
        return TimeSpectrumSet(
            x=self.x[idx],
            wavenumbers=self.wavenumbers.copy(),
            metadata=meta,
            modality=self.modality,
            label_col=self.label_col,
            group_col=self.group_col,
            batch_col=self.batch_col,
            time_col=self.time_col,
            entity_col=self.entity_col,
        )

    def groups_by_entity(self) -> Dict[str, np.ndarray]:
        if self.entity_col is None:
            raise ValueError("entity_col not set; cannot group trajectories.")
        groups: Dict[str, List[int]] = {}
        for i, ent in enumerate(self.metadata[self.entity_col].astype(str).tolist()):
            groups.setdefault(ent, []).append(i)
        return {k: np.array(v, dtype=int) for k, v in groups.items()}

    def times(self) -> np.ndarray:
        return self.metadata[self.time_col].to_numpy()

    def subset_time_window(self, t_min: float, t_max: float) -> "TimeSpectrumSet":
        if t_min > t_max:
            raise ValueError("t_min must be <= t_max.")
        mask = (self.metadata[self.time_col] >= t_min) & (self.metadata[self.time_col] <= t_max)
        idx = np.where(mask.to_numpy())[0]
        meta = self.metadata.iloc[idx].reset_index(drop=True)
        return TimeSpectrumSet(
            x=self.x[idx],
            wavenumbers=self.wavenumbers.copy(),
            metadata=meta,
            modality=self.modality,
            label_col=self.label_col,
            group_col=self.group_col,
            batch_col=self.batch_col,
            time_col=self.time_col,
            entity_col=self.entity_col,
        )

    def to_panel(self) -> pd.DataFrame:
        """Return a long-format DataFrame with time and intensities per wavenumber.

        Columns: [entity, time, wavenumber, intensity]
        """
        if self.entity_col is None:
            raise ValueError("entity_col not set; cannot build panel.")
        ent = self.metadata[self.entity_col].astype(str).to_numpy()
        t = self.metadata[self.time_col].to_numpy()
        wn = self.wavenumbers
        records: List[Tuple[str, float, float, float]] = []
        for i in range(self.x.shape[0]):
            row = self.x[i]
            for j, w in enumerate(wn):
                records.append((ent[i], float(t[i]), float(w), float(row[j])))
        return pd.DataFrame(records, columns=["entity", "time", "wavenumber", "intensity"])


__all__ = ["TimeSpectrumSet"]
