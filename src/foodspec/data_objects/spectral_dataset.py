from __future__ import annotations
"""
SpectralDataset: vendor-neutral representation and preprocessing for Raman/FTIR.

Goals:
- Represent spectra + metadata in a FAIR, instrument-aware structure.
- Provide configurable preprocessing (baseline, smoothing, normalization, spike removal).
- Peak extraction utilities compatible with the RQ engine.
- Harmonization-ready save/load to HDF5.
"""


import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF

from foodspec.features.rq import PeakDefinition

HDF5_SCHEMA_VERSION = "1.1"


def _parse_semver(version_str: Optional[str]) -> Tuple[int, int, int]:
    """Parse a semantic version string into integers.

    Args:
        version_str (str | None): Version like "1.2.3"; missing parts default
            to 0.

    Returns:
        tuple[int, int, int]: ``(major, minor, patch)``
    """
    if not version_str:
        return (0, 0, 0)
    try:
        parts = [int(p) for p in str(version_str).split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])  # type: ignore
    except Exception:
        return (0, 0, 0)


def _validate_hdf5_version(saved: Optional[str], *, allow_future: bool) -> None:
    """Validate saved HDF5 schema version compatibility.

    Args:
        saved (str | None): Version string stored in file attributes.
        allow_future (bool): If True, skip minor version checks.

    Raises:
        ValueError: If major versions differ or saved minor is newer and
            ``allow_future`` is False.
    """
    if saved is None or allow_future:
        return
    s_major, s_minor, _ = _parse_semver(saved)
    e_major, e_minor, _ = _parse_semver(HDF5_SCHEMA_VERSION)
    if s_major != e_major:
        raise ValueError(
            f"Incompatible HDF5 schema version {saved}; expected {HDF5_SCHEMA_VERSION}. "
            "Re-export the dataset with the current FoodSpec version."
        )
    if s_minor > e_minor:
        raise ValueError(
            f"HDF5 schema version {saved} is newer than supported {HDF5_SCHEMA_VERSION}. "
            "Re-export with current FoodSpec or use allow_future=True."
        )


# -------------------- Baselines --------------------
def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least squares (ALS) baseline estimation.

    Args:
        y (np.ndarray): 1D signal.
        lam (float): Smoothness penalty lambda.
        p (float): Asymmetry parameter (0-1).
        niter (int): Iterations.

    Returns:
        np.ndarray: Estimated baseline, same shape as ``y``.

    Examples:
        >>> import numpy as np
        >>> y = np.sin(np.linspace(0, 1, 10)) + 0.5
        >>> base = baseline_als(y)
        >>> base.shape
        (10,)
    """
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
    """Rubberband baseline via convex hull interpolation.

    Args:
        x (np.ndarray): Wavenumber axis (1D, increasing).
        y (np.ndarray): Signal values.

    Returns:
        np.ndarray: Baseline values along ``x``.
    """
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
    """Polynomial baseline fit.

    Args:
        x (np.ndarray): Wavenumbers.
        y (np.ndarray): Signal.
        order (int): Polynomial order.

    Returns:
        np.ndarray: Baseline values.
    """
    coeffs = np.polyfit(x, y, order)
    return np.polyval(coeffs, x)


# -------------------- Smoothing --------------------
def smooth_signal(y: np.ndarray, method: str = "savgol", window: int = 7, polyorder: int = 3) -> np.ndarray:
    """Smooth a 1D signal using Savitzky–Golay or moving average.

    Args:
        y (np.ndarray): 1D signal.
        method (str): "savgol" | "moving_average" | "none".
        window (int): Window length; odd for Savitzky–Golay.
        polyorder (int): Polynomial order for Savitzky–Golay.

    Returns:
        np.ndarray: Smoothed signal.
    """
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
    """Normalize spectra matrix row-wise by various strategies.

    Args:
        X (np.ndarray): Matrix shape (n_samples, n_points).
        mode (str): "none", "reference", "vector", "area", or "max".
        wn (np.ndarray): Wavenumber axis.
        ref_wn (float): Reference wavenumber for "reference" mode.

    Returns:
        np.ndarray: Normalized matrix of same shape as ``X``.
    """
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
    """Remove cosmic-ray spikes using a rolling median and MAD threshold.

    Args:
        y (np.ndarray): 1D signal.
        zscore_thresh (float): Threshold on |diff|/MAD to identify spikes.

    Returns:
        np.ndarray: Cleaned signal with spikes replaced by local median.
    """
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
    """Declarative preprocessing configuration for `SpectralDataset`.

    Fields mirror preprocessing steps: baseline removal, smoothing,
    normalization, spike removal, peak extraction and grid alignment.
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
    """Matrix-form spectra with aligned metadata and instrument info.

    Attributes:
        wavenumbers (np.ndarray): Axis values (cm^-1).
        spectra (np.ndarray): Shape (n_samples, n_points) intensities.
        metadata (pd.DataFrame): Sample annotations.
        instrument_meta (dict): Instrument/protocol metadata.
        logs (list[str]): Operation logs.
        history (list[dict]): Preprocessing steps applied.
    """
    wavenumbers: np.ndarray
    spectra: np.ndarray  # shape (n_samples, n_points)
    metadata: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    instrument_meta: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "SpectralDataset":
        """Deep copy of dataset arrays, metadata, and meta fields.

        Returns:
            SpectralDataset: Independent copy.
        """
        return SpectralDataset(
            wavenumbers=self.wavenumbers.copy(),
            spectra=self.spectra.copy(),
            metadata=self.metadata.copy(),
            instrument_meta=self.instrument_meta.copy(),
            logs=list(self.logs),
            history=list(self.history),
        )

    def preprocess(self, options: PreprocessingConfig) -> "SpectralDataset":
        """Apply configured preprocessing to spectra.

        Args:
            options (PreprocessingConfig): Pipeline options.

        Returns:
            SpectralDataset: New dataset with processed spectra.

        Examples:
            >>> import numpy as np, pandas as pd
            >>> ds = SpectralDataset(np.arange(3.), np.ones((2,3)), pd.DataFrame())
            >>> cfg = PreprocessingConfig(baseline_method="none", smoothing_method="none", normalization="none", spike_removal=False)
            >>> ds2 = ds.preprocess(cfg)
            >>> ds2.spectra.shape
            (2, 3)
        """
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
        """Extract peak features into a wide DataFrame.

        Args:
            peaks (Iterable[PeakDefinition]): Peak definitions with names,
                windows, and modes.

        Returns:
            pandas.DataFrame: `metadata` columns plus one column per peak.
        """
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
                vals = trapezoid(region, wn[mask], axis=1)
            else:
                max_idx = np.nanargmax(region, axis=1)
                vals = region[np.arange(region.shape[0]), max_idx]
            out[peak.name] = vals
        return out

    def save_hdf5(self, path: Union[str, Path]):
        """Save dataset to HDF5 with schema versioning and legacy keys.

        Args:
            path (str | Path): Destination file path.

        Raises:
            ImportError: If `h5py` is not installed.
        """
        if h5py is None:
            raise ImportError("h5py not installed.")
        path = Path(path)
        with h5py.File(path, "w") as f:
            f.attrs["foodspec_hdf5_schema_version"] = HDF5_SCHEMA_VERSION
            # NeXus-inspired groups
            grp_spec = f.create_group("spectra")
            ds_wn = grp_spec.create_dataset("wn_axis", data=self.wavenumbers)
            ds_wn.attrs["units"] = "cm^-1"
            grp_spec.create_dataset("intensities", data=self.spectra)
            grp_spec.create_dataset("sample_table", data=np.bytes_(self.metadata.to_json()))
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

            # Legacy datasets for backward compatibility (avoid clashing with /spectra group)
            meta_json = self.metadata.to_json()
            f.create_dataset("wavenumbers_legacy", data=self.wavenumbers)
            f.create_dataset("spectra_legacy", data=self.spectra)
            f.create_dataset("metadata_json", data=np.bytes_(meta_json))
            inst_json = json.dumps(self.instrument_meta)
            f.create_dataset("instrument_meta", data=np.bytes_(inst_json))
            f.create_dataset("logs", data=np.bytes_("\n".join(self.logs)))
            f.create_dataset("history_json", data=np.bytes_(json.dumps(self.history)))
            prov = {
                "protocol_name": self.instrument_meta.get("protocol_name"),
                "protocol_version": self.instrument_meta.get("protocol_version"),
                "harmonization_params": self.instrument_meta.get("harmonization_params"),
            }
            f.attrs["provenance_json"] = json.dumps(prov)

    @staticmethod
    def from_hdf5(path: Union[str, Path], *, allow_future: bool = False) -> "SpectralDataset":
        """Load dataset from HDF5, supporting legacy layout.

        Args:
            path (str | Path): HDF5 file path.
            allow_future (bool): If True, tolerate newer minor schema.

        Returns:
            SpectralDataset: Reconstructed dataset.

        Raises:
            ImportError: If `h5py` is not installed.
            ValueError: If schema version incompatible.
        """
        if h5py is None:
            raise ImportError("h5py not installed.")
        import io as io_module
        with h5py.File(path, "r") as f:
            _validate_hdf5_version(f.attrs.get("foodspec_hdf5_schema_version", "1.0"), allow_future=allow_future)
            if "spectra" in f and "wn_axis" in f["spectra"]:
                wn = f["spectra"]["wn_axis"][:]
                spectra = f["spectra"]["intensities"][:]
                sample_table_json = f["spectra"]["sample_table"][()].decode()
                metadata = pd.read_json(io_module.StringIO(sample_table_json))
                instrument_meta = dict(f["instrument"].attrs)
                logs = f["logs"][()].decode().splitlines() if "logs" in f else []
                history = json.loads(f["preprocessing"].attrs.get("history_json", "[]"))
            else:
                legacy_wn_key = "wavenumbers" if "wavenumbers" in f else "wavenumbers_legacy"
                legacy_spec_key = "spectra" if "spectra" in f else "spectra_legacy"
                wn = f[legacy_wn_key][:]
                spectra = f[legacy_spec_key][:]
                metadata_json = f["metadata_json"][()].decode()
                metadata = pd.read_json(io_module.StringIO(metadata_json)) if metadata_json.strip() else pd.DataFrame()
                instrument_meta = json.loads(f["instrument_meta"][()].decode())
                logs = f["logs"][()].decode().splitlines()
                history_json = f["history_json"][()].decode() if "history_json" in f else "[]"
                history = json.loads(history_json) if history_json.strip() else []
                if "provenance_json" in f.attrs:
                    try:
                        prov = json.loads(f.attrs["provenance_json"])
                        instrument_meta.update({k: v for k, v in prov.items() if v is not None})
                    except Exception:
                        pass
        return SpectralDataset(wn, spectra, metadata, instrument_meta, logs, history)


def harmonize_datasets(
    datasets: List[SpectralDataset],
    target_wavenumbers: Optional[np.ndarray] = None,
) -> List[SpectralDataset]:
    """Interpolate all spectra to a common wavenumber grid.

    Args:
        datasets (list[SpectralDataset]): Input datasets.
        target_wavenumbers (np.ndarray | None): Target axis; if None, uses the
            longest grid among inputs.

    Returns:
        list[SpectralDataset]: Harmonized datasets with shared grid.
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
    """Hyperspectral cube flattened to spectra with spatial tracking.

    The cube has shape (y, x, wn) but is stored as spectra with shape
    (n_pixels, wn); spatial dimensions are given by `shape_xy`.
    """

    shape_xy: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    label_map: Optional[np.ndarray] = None  # segmentation labels
    roi_masks: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def from_cube(
        cube: np.ndarray,
        wavenumbers: np.ndarray,
        metadata: pd.DataFrame,
        instrument_meta: Dict[str, Any] = None,
    ) -> "HyperspectralDataset":
        """Construct flattened dataset from a hyperspectral cube.

        Args:
            cube (np.ndarray): Array shape (y, x, wn).
            wavenumbers (np.ndarray): Axis values.
            metadata (pd.DataFrame): Sample/site metadata.
            instrument_meta (dict | None): Instrument details.

        Returns:
            HyperspectralDataset: Flattened dataset with `shape_xy` set.
        """
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
        """Reshape spectra back to (y, x, wn) cube using `shape_xy`.

        Returns:
            np.ndarray: Hyperspectral cube.
        """
        y, x = self.shape_xy
        return self.spectra.reshape((y, x, -1))

    def segment(self, method: str = "kmeans", n_clusters: int = 3) -> np.ndarray:
        """Segment pixels into clusters using kmeans, hierarchical, or NMF.

        Args:
            method (str): "kmeans" | "hierarchical" | "nmf".
            n_clusters (int): Number of clusters/components.

        Returns:
            np.ndarray: Label map shape `shape_xy`.

        Raises:
            ValueError: If method unknown.
        """
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
        """Average spectrum from a binary ROI mask.

        Args:
            mask (np.ndarray): Boolean/int array of shape `shape_xy`.

        Returns:
            SpectralDataset: Single-row dataset with the average ROI spectrum.

        Raises:
            ValueError: If `mask` shape mismatches `shape_xy`.
        """
        cube = self.to_cube()
        if mask.shape != self.shape_xy:
            raise ValueError("Mask shape does not match cube")
        mask_bool = mask.astype(bool)
        spectra = cube[mask_bool]
        avg = np.nanmean(spectra, axis=0, keepdims=True)
        meta = pd.DataFrame({"roi_pixels": [mask_bool.sum()]})
        return SpectralDataset(self.wavenumbers.copy(), avg, meta, self.instrument_meta.copy(), list(self.logs))

    def unmix_mcr_als(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int = 0,
    ) -> Dict[str, Any]:
        """Unmix hyperspectral data using MCR-ALS."""
        from foodspec.hyperspectral.unmixing import mcr_als

        C, S, info = mcr_als(
            self.spectra,
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        y, x = self.shape_xy
        if y * x != C.shape[0]:
            raise ValueError("shape_xy does not match spectra length.")
        abundances = C.reshape((y, x, n_components))
        return {"abundances": abundances, "components": S, "info": info}

    def save_hdf5(self, path: Union[str, Path]):
        """Save hyperspectral cube to HDF5, including ROI artifacts if present.

        Args:
            path (str | Path): Destination file path.

        Raises:
            ImportError: If `h5py` is not installed.
        """
        if h5py is None:
            raise ImportError("h5py not installed.")
        path = Path(path)
        cube = self.to_cube()
        with h5py.File(path, "w") as f:
            f.attrs["foodspec_hdf5_schema_version"] = HDF5_SCHEMA_VERSION
            grp_spec = f.create_group("spectra")
            ds_wn = grp_spec.create_dataset("wn_axis", data=self.wavenumbers)
            ds_wn.attrs["units"] = "cm^-1"
            grp_spec.create_dataset("cube", data=cube)
            grp_spec.create_dataset("shape_xy", data=np.array(self.shape_xy))
            grp_spec.create_dataset("sample_table", data=np.bytes_(self.metadata.to_json()))
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
            f.create_dataset("metadata_json", data=np.bytes_(meta_json))
            inst_json = json.dumps(self.instrument_meta)
            f.create_dataset("instrument_meta", data=np.bytes_(inst_json))
            f.create_dataset("logs", data=np.bytes_("\n".join(self.logs)))
            f.create_dataset("history_json", data=np.bytes_(json.dumps(self.history)))

    @staticmethod
    def from_hdf5(path: Union[str, Path], *, allow_future: bool = False) -> "HyperspectralDataset":
        """Load hyperspectral dataset from HDF5, supporting legacy layout.

        Args:
            path (str | Path): HDF5 path.
            allow_future (bool): If True, tolerate newer minor schema.

        Returns:
            HyperspectralDataset: Reconstructed dataset with `shape_xy` and
            optional ROI assets.

        Raises:
            ImportError: If `h5py` is not installed.
            ValueError: If schema version incompatible.
        """
        if h5py is None:
            raise ImportError("h5py not installed.")
        with h5py.File(path, "r") as f:
            _validate_hdf5_version(f.attrs.get("foodspec_hdf5_schema_version", "1.0"), allow_future=allow_future)
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


__all__ = ["SpectralDataset", "HyperspectralDataset", "PreprocessingConfig", "HDF5_SCHEMA_VERSION"]
