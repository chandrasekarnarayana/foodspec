"""Data ingestion and standardization registry for FoodSpec (Phase 2).

Provides registry-driven loaders, I/O quality metrics, and validation checks:
- CSV/TXT parsing with delimiter inference
- Folder crawler with pattern match
- Vendor export adapters (JCAMP, SPC, OPUS)
- HDF5 loader/writer (lightweight schema)
- Spectral cube loader (HSI) v2 (minimal placeholder)
- Metrics: % files parsed, % spectra valid/resampled/flagged
- Validation: monotonic axis, uniform spacing metric, axis overlap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.io.core import detect_format, read_spectra
from foodspec.io.text_formats import read_jcamp
from foodspec.io.vendor_formats import read_opus, read_spc
from foodspec.plugins import ensure_plugins_loaded
from foodspec.plugins.loaders import vendor_loader_registry

try:  # Optional dependency
    import h5py
except Exception:  # pragma: no cover - optional
    h5py = None

LoaderFn = Callable[..., "IngestResult"]


@dataclass
class IngestResult:
    """Container for ingestion outputs and quality metrics."""

    dataset: FoodSpectrumSet
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class IORegistry:
    """Simple registry pattern for ingestion algorithms."""

    def __init__(self) -> None:
        self._loaders: Dict[str, LoaderFn] = {}

    def register(self, name: str, fn: LoaderFn) -> None:
        if name in self._loaders:
            raise ValueError(f"Loader '{name}' already registered")
        self._loaders[name] = fn

    def load(self, name: str, /, *args, **kwargs) -> IngestResult:
        if name not in self._loaders:
            raise KeyError(f"Loader '{name}' not found; available: {sorted(self._loaders)}")
        return self._loaders[name](*args, **kwargs)

    def available(self) -> List[str]:
        return sorted(self._loaders)


def _infer_delimiter(path: Path) -> str:
    """Infer delimiter by sampling the first line."""

    candidates = [",", "\t", ";", " "]
    with path.open("r", errors="ignore") as f:
        first_line = f.readline()
    counts = {delim: first_line.count(delim) for delim in candidates}
    return max(counts, key=counts.get) or ","


def _read_csv_or_txt(path: Path, modality: str = "raman") -> FoodSpectrumSet:
    delim = _infer_delimiter(path)
    df = pd.read_csv(path, sep=delim, engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"File {path} must contain at least two columns (wavenumber + intensity).")

    # Wide format if more than two columns (multiple spectra)
    if df.shape[1] > 2 or df.shape[0] <= df.shape[1]:
        wavenumbers = df.iloc[:, 0].to_numpy(dtype=float)
        spectra = df.iloc[:, 1:].to_numpy(dtype=float).T
        metadata = pd.DataFrame({"sample_id": df.columns[1:]})
    else:
        # Long format: two columns [wavenumber, intensity]
        wavenumbers = df.iloc[:, 0].to_numpy(dtype=float)
        spectra = df.iloc[:, 1].to_numpy(dtype=float)[None, :]
        metadata = pd.DataFrame({"sample_id": [path.stem]})

    return FoodSpectrumSet(x=spectra, wavenumbers=wavenumbers, metadata=metadata, modality=modality)


def _read_folder(
    folder: Path,
    pattern: str = "*.txt",
    modality: str = "raman",
) -> Tuple[FoodSpectrumSet, Dict[str, Any]]:
    files = sorted(folder.glob(pattern))
    if not files:
        raise ValueError(f"No files matching pattern '{pattern}' in {folder}")

    w_axes: List[np.ndarray] = []
    spectra: List[np.ndarray] = []
    sample_ids: List[str] = []
    parsed_files = 0
    flagged = 0
    resampled = 0

    for f in files:
        try:
            fs = _read_csv_or_txt(f, modality=modality)
            w_axes.append(fs.wavenumbers)
            spectra.append(fs.x.squeeze())
            sample_ids.append(f.stem)
            parsed_files += 1
        except Exception:
            flagged += 1
            continue

    if not spectra:
        raise ValueError(f"No spectra parsed in {folder}")

    reference = w_axes[0]
    stacked = []
    for wav, spec in zip(w_axes, spectra):
        if wav.shape != reference.shape or not np.allclose(wav, reference):
            interp = np.interp(reference, wav, spec, left=np.nan, right=np.nan)
            stacked.append(interp)
            resampled += 1
        else:
            stacked.append(spec)
    x = np.vstack(stacked)
    metadata = pd.DataFrame({"sample_id": sample_ids})

    metrics = {
        "total_files": len(files),
        "parsed_files": parsed_files,
        "flagged_files": flagged,
        "resampled_spectra": resampled,
    }

    dataset = FoodSpectrumSet(x=x, wavenumbers=reference, metadata=metadata, modality=modality)
    return dataset, metrics


def _compute_axis_metrics(dataset: FoodSpectrumSet) -> Dict[str, Any]:
    wn = dataset.wavenumbers
    diffs = np.diff(wn)
    monotonic = bool(np.all(diffs > 0))
    nonuniform = float(np.std(diffs) / max(np.mean(diffs), 1e-9))
    coverage = {
        "min": float(np.min(wn)),
        "max": float(np.max(wn)),
        "range": float(np.max(wn) - np.min(wn)),
    }
    return {
        "monotonic_axis": monotonic,
        "grid_nonuniformity": nonuniform,
        "axis_coverage": coverage,
    }


def _finalize_metrics(raw: Dict[str, Any], dataset: FoodSpectrumSet) -> Dict[str, Any]:
    total = max(raw.get("total_files", 1), 1)
    spectra = dataset.x.shape[0]
    parsed_files = raw.get("parsed_files", spectra)
    flagged = raw.get("flagged_files", 0)
    resampled = raw.get("resampled_spectra", 0)

    metrics = {
        **raw,
        "total_spectra": spectra,
        "parsed_pct": 100.0 * parsed_files / total,
        "flagged_pct": 100.0 * flagged / total,
        "resampled_pct": 100.0 * resampled / max(spectra, 1),
    }
    metrics.update(_compute_axis_metrics(dataset))
    return metrics


def load_csv_or_txt(path: str | Path, modality: str = "raman") -> IngestResult:
    p = Path(path)
    dataset = _read_csv_or_txt(p, modality=modality)
    raw_metrics = {
        "total_files": 1,
        "parsed_files": 1,
        "flagged_files": 0,
        "resampled_spectra": 0,
    }
    metrics = _finalize_metrics(raw_metrics, dataset)
    return IngestResult(dataset=dataset, metrics=metrics, diagnostics={"source": str(p)})


def load_folder_pattern(path: str | Path, pattern: str = "*.txt", modality: str = "raman") -> IngestResult:
    dataset, raw = _read_folder(Path(path), pattern=pattern, modality=modality)
    metrics = _finalize_metrics(raw, dataset)
    return IngestResult(dataset=dataset, metrics=metrics, diagnostics={"pattern": pattern})


def _coerce_ingest_result(result: object, fmt: str, modality: str) -> IngestResult:
    if isinstance(result, IngestResult):
        return result
    if isinstance(result, FoodSpectrumSet):
        dataset = result
    elif hasattr(result, "wavenumbers") and hasattr(result, "spectra"):
        metadata = getattr(result, "metadata", pd.DataFrame())
        dataset = FoodSpectrumSet(
            x=np.asarray(getattr(result, "spectra")),
            wavenumbers=np.asarray(getattr(result, "wavenumbers")),
            metadata=metadata,
            modality=getattr(result, "modality", modality),
        )
    else:
        raise TypeError(f"Vendor loader for format '{fmt}' must return IngestResult or spectrum-like object")

    metrics = _finalize_metrics({"total_files": 1, "parsed_files": 1, "flagged_files": 0, "resampled_spectra": 0}, dataset)
    return IngestResult(dataset=dataset, metrics=metrics, diagnostics={"format": fmt})


def load_vendor(path: str | Path, format: Optional[str] = None, **kwargs) -> IngestResult:
    fmt = (format or detect_format(path) or "unknown").lower()
    ensure_plugins_loaded()
    plugin_loader = vendor_loader_registry.get(fmt)
    if plugin_loader is None and fmt == "unknown":
        suffix = Path(path).suffix.lstrip(".").lower()
        plugin_loader = vendor_loader_registry.get(suffix) if suffix else None
    if plugin_loader is not None:
        plugin_result = plugin_loader(path, **kwargs)
        return _coerce_ingest_result(plugin_result, fmt, kwargs.get("modality", "raman"))

    if fmt == "jcamp":
        ds = read_jcamp(path, **kwargs)
    elif fmt == "spc":
        ds = read_spc(path, **kwargs)
    elif fmt == "opus":
        ds = read_opus(path, **kwargs)
    else:
        raise ValueError(f"Unsupported vendor format: {fmt}")
    metrics = _finalize_metrics({"total_files": 1, "parsed_files": 1, "flagged_files": 0, "resampled_spectra": 0}, ds)
    return IngestResult(dataset=ds, metrics=metrics, diagnostics={"format": fmt})


def load_hdf5(path: str | Path, modality: str = "raman") -> IngestResult:
    if h5py is None:
        raise ImportError("h5py is required for HDF5 loading")
    p = Path(path)
    with h5py.File(p, "r") as f:
        x = np.array(f["x"])
        w = np.array(f["wavenumbers"])
        metadata_json = f["metadata"].asstr()[()]
        meta = pd.read_json(metadata_json)
    ds = FoodSpectrumSet(x=x, wavenumbers=w, metadata=meta, modality=modality)
    metrics = _finalize_metrics({"total_files": 1, "parsed_files": 1, "flagged_files": 0, "resampled_spectra": 0}, ds)
    return IngestResult(dataset=ds, metrics=metrics, diagnostics={"format": "hdf5"})


def save_hdf5(dataset: FoodSpectrumSet, path: str | Path) -> Path:
    if h5py is None:
        raise ImportError("h5py is required for HDF5 writing")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = dataset.metadata.to_json()
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=dataset.x)
        f.create_dataset("wavenumbers", data=dataset.wavenumbers)
        f.create_dataset("metadata", data=np.string_(metadata_json))
    return p


def load_spectral_cube(path: str | Path, wavenumbers: Optional[np.ndarray] = None, modality: str = "hsi") -> IngestResult:
    p = Path(path)
    if p.suffix.lower() in {".npz"}:
        data = np.load(p)["cube"]
    elif p.suffix.lower() in {".npy"}:
        data = np.load(p)
    else:
        raise ValueError("Spectral cube loader currently supports .npy or .npz with 'cube' key")
    if data.ndim != 3:
        raise ValueError("Spectral cube must be 3D (rows, cols, bands)")
    rows, cols, bands = data.shape
    cube_reshaped = data.reshape(rows * cols, bands)
    if wavenumbers is None:
        wavenumbers = np.linspace(0, bands - 1, bands)
    metadata = pd.DataFrame({
        "row": np.repeat(np.arange(rows), cols),
        "col": np.tile(np.arange(cols), rows),
    })
    ds = FoodSpectrumSet(x=cube_reshaped, wavenumbers=wavenumbers, metadata=metadata, modality=modality)
    raw = {
        "total_files": 1,
        "parsed_files": 1,
        "flagged_files": 0,
        "resampled_spectra": 0,
    }
    metrics = _finalize_metrics(raw, ds)
    diagnostics = {"shape": list(data.shape)}
    return IngestResult(dataset=ds, metrics=metrics, diagnostics=diagnostics)


# Default registry with built-in loaders
DEFAULT_IO_REGISTRY = IORegistry()
DEFAULT_IO_REGISTRY.register("csv", load_csv_or_txt)
DEFAULT_IO_REGISTRY.register("folder", load_folder_pattern)
DEFAULT_IO_REGISTRY.register("vendor", load_vendor)
DEFAULT_IO_REGISTRY.register("hdf5", load_hdf5)
DEFAULT_IO_REGISTRY.register("spectral_cube", load_spectral_cube)
DEFAULT_IO_REGISTRY.register("auto", lambda path, **kwargs: _auto_loader(path, **kwargs))


def _auto_loader(path: str | Path, **kwargs) -> IngestResult:
    fmt = detect_format(path)
    if Path(path).is_dir():
        return load_folder_pattern(path, **kwargs)
    if fmt in {"csv", "txt"}:
        return load_csv_or_txt(path, **kwargs)
    if fmt in {"jcamp", "spc", "opus"}:
        return load_vendor(path, format=fmt, **kwargs)
    if fmt == "unknown":
        # Fallback to generic core reader
        ds = read_spectra(path, **kwargs)
        metrics = _finalize_metrics({"total_files": 1, "parsed_files": 1, "flagged_files": 0, "resampled_spectra": 0}, ds)
        return IngestResult(dataset=ds, metrics=metrics, diagnostics={"format": "auto"})
    return load_csv_or_txt(path, **kwargs)
