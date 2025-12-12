"""
SpectralDataset: vendor-neutral representation and preprocessing for Raman/FTIR.

Goals:
- Represent spectra + metadata in a FAIR, instrument-aware structure.
- Provide configurable preprocessing (baseline, smoothing, normalization, spike removal).
- Peak extraction utilities compatible with the RQ engine.
- Harmonization-ready save/load to HDF5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None

from foodspec.rq import PeakDefinition
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import NMF


# -------------------- Baselines --------------------
def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)


def baseline_rubberband(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Simple convex hull rubberband
    v = np.array([x, y]).T
    hull = [0]
    for i in range(1, len(v) - 1):
        while len(hull) >= 2:
            p1, p2 = v[hull[-2]], v[hull[-1]]
            cross = (p2[0] - p1[0]) * (v[i][1] - p1[1]) - (p2[1] - p1[1]) * (v[i][0] - p1[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    hull.append(len(v) - 1)
    hull_x = v[hull][:, 0]
    hull_y = v[hull][:, 1]
    return np.interp(x, hull_x, hull_y)


def baseline_polynomial(x: np.ndarray, y: np.ndarray, order: int = 3) -> np.ndarray:
    coeffs = np.polyfit(x, y, order)
    return np.polyval(coeffs, x)


# -------------------- Smoothing --------------------
def smooth_signal(y: np.ndarray, method: str = "savgol", window: int = 7, polyorder: int = 3) -> np.ndarray:
    if method == "savgol":
        window = max(3, window if window % 2 == 1 else window + 1)
        return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")
    if method == "moving_average":
        win = max(3, window)
        pad = win // 2
        padded = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(win) / win
        return np.convolve(padded, kernel, mode="valid")
    return y


# -------------------- Normalization --------------------
def normalize_matrix(X: np.ndarray, mode: str, wn: np.ndarray, ref_wn: float) -> np.ndarray:
    if mode == "none":
        return X
    if mode == "reference":
        ref_idx = int(np.argmin(np.abs(wn - ref_wn)))
        norms = X[:, [ref_idx]] + 1e-12
        return X / norms
    if mode == "vector":
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms
    if mode == "area":
        norms = np.sum(X, axis=1, keepdims=True) + 1e-12
        return X / norms
    if mode == "max":
        norms = np.max(X, axis=1, keepdims=True) + 1e-12
        return X / norms
    return X


# -------------------- Spike / cosmic-ray removal --------------------
def remove_spikes(y: np.ndarray, zscore_thresh: float = 8.0) -> np.ndarray:
    diff = y - pd.Series(y).rolling(5, center=True, min_periods=1).median().to_numpy()
    mad = np.median(np.abs(diff)) + 1e-12
    z = np.abs(diff) / mad
    spike_mask = z > zscore_thresh
    if not spike_mask.any():
        return y
    y_clean = y.copy()
    median = pd.Series(y).rolling(5, center=True, min_periods=1).median().to_numpy()
    y_clean[spike_mask] = median[spike_mask]
    return y_clean


@dataclass
class PreprocessingConfig:
    """
    Declarative preprocessing configuration.
    """
    baseline_method: str = "als"  # als | rubberband | polynomial | none
    baseline_enabled: bool = True
    baseline_lambda: float = 1e5
    baseline_p: float = 0.01
    baseline_order: int = 3  # for polynomial
    smoothing_method: str = "savgol"  # savgol | moving_average | none
    smooth_enabled: bool = True
    smoothing_window: int = 7
    smoothing_polyorder: int = 3
    normalization: str = "reference"  # reference | vector | area | max | none
    reference_wavenumber: float = 2720.0
    spike_removal: bool = True
    spike_zscore_thresh: float = 8.0
    peak_strategy: str = "max"  # max | area
    peak_definitions: List[PeakDefinition] = field(default_factory=list)
    align_to_common_grid: bool = False
    target_grid: Optional[List[float]] = None
    interp_method: str = "interp"


@dataclass
class SpectralDataset:
    wavenumbers: np.ndarray
    spectra: np.ndarray  # shape (n_samples, n_points)
    metadata: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    instrument_meta: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "SpectralDataset":
        return SpectralDataset(
            wavenumbers=self.wavenumbers.copy(),
            spectra=self.spectra.copy(),
            metadata=self.metadata.copy(),
            instrument_meta=self.instrument_meta.copy(),
            logs=list(self.logs),
            history=list(self.history),
        )

    def preprocess(self, options: PreprocessingConfig) -> "SpectralDataset":
        ds = self.copy()
        wn = ds.wavenumbers
        X = ds.spectra.copy()
        ds.logs.append(f"preprocess: {options}")
        ds.history.append({"step": "preprocess", "params": asdict(options)})
        # Spike removal
        if options.spike_removal:
            X = np.apply_along_axis(remove_spikes, 1, X, options.spike_zscore_thresh)
        # Baseline
        for i in range(X.shape[0]):
            y = X[i]
            if options.baseline_method == "als":
                base = baseline_als(y, lam=options.baseline_lambda, p=options.baseline_p)
            elif options.baseline_method == "rubberband":
                base = baseline_rubberband(wn, y)
            elif options.baseline_method == "polynomial":
                base = baseline_polynomial(wn, y, order=options.baseline_order)
            else:
                base = np.zeros_like(y)
            X[i] = y - base
        # Smoothing
        if options.smoothing_method != "none":
            X = np.apply_along_axis(
                smooth_signal, 1, X, options.smoothing_method, options.smoothing_window, options.smoothing_polyorder
            )
        # Normalization
        X = normalize_matrix(X, options.normalization, wn, options.reference_wavenumber)
        ds.spectra = X
        return ds

    def to_peaks(self, peaks: Iterable[PeakDefinition]) -> pd.DataFrame:
        peak_defs = list(peaks)
        out = self.metadata.copy()
        wn = self.wavenumbers
        for peak in peak_defs:
            if peak.wavenumber is None:
                continue
            center = float(peak.wavenumber)
            low, high = peak.window if peak.window else (center - 10, center + 10)
            mask = (wn >= low) & (wn <= high)
            if not mask.any():
                out[peak.name] = np.nan
                continue
            region = self.spectra[:, mask]
            if peak.mode == "area":
                vals = np.trapz(region, wn[mask], axis=1)
            else:
                max_idx = np.nanargmax(region, axis=1)
                vals = region[np.arange(region.shape[0]), max_idx]
            out[peak.name] = vals
        return out

    def save_hdf5(self, path: Union[str, Path]):
        if h5py is None:
            raise ImportError("h5py not installed.")
        path = Path(path)
        with h5py.File(path, "w") as f:
            f.attrs["foodspec_hdf5_schema_version"] = "1.0"
            # NeXus-inspired groups
            grp_spec = f.create_group("spectra")
            ds_wn = grp_spec.create_dataset("wn_axis", data=self.wavenumbers)
            ds_wn.attrs["units"] = "cm^-1"
            grp_spec.create_dataset("intensities", data=self.spectra)
            grp_spec.create_dataset("sample_table", data=np.string_(self.metadata.to_json()))
            grp_inst = f.create_group("instrument")
            for k, v in self.instrument_meta.items():
                try:
                    grp_inst.attrs[k] = v
                except Exception:
                    grp_inst.attrs[k] = str(v)
            grp_pre = f.create_group("preprocessing")
            grp_pre.attrs["history_json"] = json.dumps(self.history)
            grp_proto = f.create_group("protocol")
            grp_proto.attrs["name"] = self.instrument_meta.get("protocol_name", "")
            grp_proto.attrs["version"] = self.instrument_meta.get("protocol_version", "")
            grp_proto.attrs["steps_json"] = json.dumps(self.instrument_meta.get("protocol_steps", []))

            # Legacy datasets for backward compatibility
            f.create_dataset("wavenumbers", data=self.wavenumbers)
            f.create_dataset("spectra", data=self.spectra)
            meta_json = self.metadata.to_json()
            f.create_dataset("metadata_json", data=np.string_(meta_json))
            inst_json = json.dumps(self.instrument_meta)
            f.create_dataset("instrument_meta", data=np.string_(inst_json))
            f.create_dataset("logs", data=np.string_("\n".join(self.logs)))
            f.create_dataset("history_json", data=np.string_(json.dumps(self.history)))
            prov = {
                "protocol_name": self.instrument_meta.get("protocol_name"),
                "protocol_version": self.instrument_meta.get("protocol_version"),
                "harmonization_params": self.instrument_meta.get("harmonization_params"),
            }
            f.attrs["provenance_json"] = json.dumps(prov)

    @staticmethod
    def from_hdf5(path: Union[str, Path]) -> "SpectralDataset":
        if h5py is None:
            raise ImportError("h5py not installed.")
        with h5py.File(path, "r") as f:
            schema_version = f.attrs.get("foodspec_hdf5_schema_version", "0")
            if "spectra" in f and "wn_axis" in f["spectra"]:
                wn = f["spectra"]["wn_axis"][:]
                spectra = f["spectra"]["intensities"][:]
                metadata = pd.read_json(f["spectra"]["sample_table"][()].decode())
                instrument_meta = dict(f["instrument"].attrs)
                logs = f["logs"][()].decode().splitlines() if "logs" in f else []
                history = json.loads(f["preprocessing"].attrs.get("history_json", "[]"))
            else:
                wn = f["wavenumbers"][:]
                spectra = f["spectra"][:]
                metadata = pd.read_json(f["metadata_json"][()].decode())
                instrument_meta = json.loads(f["instrument_meta"][()].decode())
                logs = f["logs"][()].decode().splitlines()
                history = json.loads(f["history_json"][()].decode()) if "history_json" in f else []
                if "provenance_json" in f.attrs:
                    try:
                        prov = json.loads(f.attrs["provenance_json"])
                        instrument_meta.update({k: v for k, v in prov.items() if v is not None})
                    except Exception:
                        pass
        return SpectralDataset(wn, spectra, metadata, instrument_meta, logs, history)


def harmonize_datasets(datasets: List[SpectralDataset], target_wavenumbers: Optional[np.ndarray] = None) -> List[SpectralDataset]:
    """
    Simple harmonization: interpolate all spectra to a common wavenumber grid.
    If target_wavenumbers is None, use the median grid of the inputs.
    """
    if target_wavenumbers is None:
        # choose grid from longest spectrum
        target_wavenumbers = max(datasets, key=lambda d: len(d.wavenumbers)).wavenumbers
    harmonized = []
    for ds in datasets:
        wn = ds.wavenumbers
        X = ds.spectra
        interp = np.vstack([np.interp(target_wavenumbers, wn, row) for row in X])
        ds_h = ds.copy()
        ds_h.wavenumbers = target_wavenumbers.copy()
        ds_h.spectra = interp
        ds_h.logs.append(f"harmonized to grid len={len(target_wavenumbers)}")
        harmonized.append(ds_h)
    return harmonized


@dataclass
class HyperspectralDataset(SpectralDataset):
    """
    Hyperspectral cube: shape (y, x, wn). Stored as spectra flattened to (n_pixels, wn)
    with spatial shape tracked separately.
    """
    shape_xy: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    label_map: Optional[np.ndarray] = None  # segmentation labels
    roi_masks: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def from_cube(cube: np.ndarray, wavenumbers: np.ndarray, metadata: pd.DataFrame, instrument_meta: Dict[str, Any] = None) -> "HyperspectralDataset":
        y, x, wn_len = cube.shape
        spectra = cube.reshape((-1, wn_len))
        return HyperspectralDataset(
            wavenumbers=wavenumbers,
            spectra=spectra,
            metadata=metadata,
            instrument_meta=instrument_meta or {},
            shape_xy=(y, x),
        )

    def to_cube(self) -> np.ndarray:
        y, x = self.shape_xy
        return self.spectra.reshape((y, x, -1))

    def segment(self, method: str = "kmeans", n_clusters: int = 3) -> np.ndarray:
        X = self.spectra
        if method == "kmeans":
            labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(X)
        elif method == "hierarchical":
            labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
        elif method == "nmf":
            model = NMF(n_components=n_clusters, init="random", random_state=0, max_iter=500)
            W = model.fit_transform(np.maximum(X, 0))
            labels = np.argmax(W, axis=1)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        self.label_map = labels.reshape(self.shape_xy)
        return labels.reshape(self.shape_xy)

    def roi_spectrum(self, mask: np.ndarray) -> SpectralDataset:
        cube = self.to_cube()
        if mask.shape != self.shape_xy:
            raise ValueError("Mask shape does not match cube")
        mask_bool = mask.astype(bool)
        spectra = cube[mask_bool]
        avg = np.nanmean(spectra, axis=0, keepdims=True)
        meta = pd.DataFrame({"roi_pixels": [mask_bool.sum()]})
        return SpectralDataset(self.wavenumbers.copy(), avg, meta, self.instrument_meta.copy(), list(self.logs))

    def save_hdf5(self, path: Union[str, Path]):
        if h5py is None:
            raise ImportError("h5py not installed.")
        path = Path(path)
        cube = self.to_cube()
        with h5py.File(path, "w") as f:
            f.attrs["foodspec_hdf5_schema_version"] = "1.0"
            grp_spec = f.create_group("spectra")
            ds_wn = grp_spec.create_dataset("wn_axis", data=self.wavenumbers)
            ds_wn.attrs["units"] = "cm^-1"
            grp_spec.create_dataset("cube", data=cube)
            grp_spec.create_dataset("shape_xy", data=np.array(self.shape_xy))
            grp_spec.create_dataset("sample_table", data=np.string_(self.metadata.to_json()))
            grp_inst = f.create_group("instrument")
            for k, v in self.instrument_meta.items():
                try:
                    grp_inst.attrs[k] = v
                except Exception:
                    grp_inst.attrs[k] = str(v)
            grp_pre = f.create_group("preprocessing")
            grp_pre.attrs["history_json"] = json.dumps(self.history)
            grp_proto = f.create_group("protocol")
            grp_proto.attrs["name"] = self.instrument_meta.get("protocol_name", "")
            grp_proto.attrs["version"] = self.instrument_meta.get("protocol_version", "")
            grp_proto.attrs["steps_json"] = json.dumps(self.instrument_meta.get("protocol_steps", []))
            if self.label_map is not None:
                f.create_dataset("label_map", data=self.label_map)
            if self.roi_masks:
                grp = f.create_group("roi_masks")
                for name, mask in self.roi_masks.items():
                    grp.create_dataset(name, data=mask.astype(np.uint8))
            prov = {
                "protocol_name": self.instrument_meta.get("protocol_name"),
                "protocol_version": self.instrument_meta.get("protocol_version"),
                "harmonization_params": self.instrument_meta.get("harmonization_params"),
            }
            f.attrs["provenance_json"] = json.dumps(prov)
            # Legacy for compatibility
            f.create_dataset("wavenumbers", data=self.wavenumbers)
            f.create_dataset("cube", data=cube)
            f.create_dataset("shape_xy", data=np.array(self.shape_xy))
            meta_json = self.metadata.to_json()
            f.create_dataset("metadata_json", data=np.string_(meta_json))
            inst_json = json.dumps(self.instrument_meta)
            f.create_dataset("instrument_meta", data=np.string_(inst_json))
            f.create_dataset("logs", data=np.string_("\n".join(self.logs)))
            f.create_dataset("history_json", data=np.string_(json.dumps(self.history)))

    @staticmethod
    def from_hdf5(path: Union[str, Path]) -> "HyperspectralDataset":
        if h5py is None:
            raise ImportError("h5py not installed.")
        with h5py.File(path, "r") as f:
            if "spectra" in f and "wn_axis" in f["spectra"]:
                wn = f["spectra"]["wn_axis"][:]
                cube = f["spectra"]["cube"][:]
                shape_xy = tuple(f["spectra"]["shape_xy"][:].astype(int))
                metadata = pd.read_json(f["spectra"]["sample_table"][()].decode())
                instrument_meta = dict(f["instrument"].attrs)
                logs = f["logs"][()].decode().splitlines() if "logs" in f else []
                history = json.loads(f["preprocessing"].attrs.get("history_json", "[]"))
                label_map = f["label_map"][:] if "label_map" in f else None
                roi_masks = {}
                if "roi_masks" in f:
                    for name, ds in f["roi_masks"].items():
                        roi_masks[name] = ds[:]
            else:
                wn = f["wavenumbers"][:]
                cube = f["cube"][:]
                shape_xy = tuple(f["shape_xy"][:].astype(int))
                metadata = pd.read_json(f["metadata_json"][()].decode())
                instrument_meta = json.loads(f["instrument_meta"][()].decode())
                logs = f["logs"][()].decode().splitlines()
                history = json.loads(f["history_json"][()].decode()) if "history_json" in f else []
                label_map = f["label_map"][:] if "label_map" in f else None
                roi_masks = {}
                if "roi_masks" in f:
                    for name, ds in f["roi_masks"].items():
                        roi_masks[name] = ds[:]
                if "provenance_json" in f.attrs:
                    try:
                        prov = json.loads(f.attrs["provenance_json"])
                        instrument_meta.update({k: v for k, v in prov.items() if v is not None})
                    except Exception:
                        pass
        spectra = cube.reshape((-1, cube.shape[-1]))
        return HyperspectralDataset(
            wavenumbers=wn,
            spectra=spectra,
            metadata=metadata,
            instrument_meta=instrument_meta,
            logs=logs,
            shape_xy=shape_xy,
            history=history,
            label_map=label_map,
            roi_masks=roi_masks if roi_masks else None,
        )

    def projection(self, mode: str = "mean", band: Optional[Tuple[float, float]] = None) -> np.ndarray:
        cube = self.to_cube()
        if mode == "mean":
            return np.nanmean(cube, axis=2)
        if mode == "band" and band:
            mask = (self.wavenumbers >= band[0]) & (self.wavenumbers <= band[1])
            return np.nanmean(cube[:, :, mask], axis=2)
        return np.nanmean(cube, axis=2)


__all__ = ["SpectralDataset", "HyperspectralDataset", "PreprocessingConfig"]
