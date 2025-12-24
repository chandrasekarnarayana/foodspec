"""Preprocessing engine with Step abstraction and AutoPreprocess (Phase 3).

Features:
- Step protocol: fit(ds), transform(ds), to_dict(), hash()
- Built-in steps for baseline, smoothing, normalization, derivatives, alignment (stub v2), resampling
- Quality metrics: baseline_residual_energy_lowfreq, negativity_fraction, smoothness_penalty,
  snr_gain, peak_preservation, scale_stability
- AutoPreprocess: search over candidate pipelines and pick best by heuristic score

Notes:
- All steps operate on FoodSpectrumSet; they return new FoodSpectrumSet copies
- Hashes are short SHA256 digests of configuration
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from foodspec.core.dataset import FoodSpectrumSet

# ----------------------------- Utilities -----------------------------


def _short_hash(data: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("Expected 2D array")
    return arr


def _clone_ds(ds: FoodSpectrumSet, x: np.ndarray, wavenumbers: Optional[np.ndarray] = None) -> FoodSpectrumSet:
    meta = ds.metadata.copy() if hasattr(ds.metadata, "copy") else ds.metadata
    return ds.__class__(x=x, wavenumbers=wavenumbers or ds.wavenumbers, metadata=meta, modality=ds.modality)


def _xcorr_shift(ref: np.ndarray, sig: np.ndarray, max_shift: int) -> int:
    """Return integer shift that best aligns sig to ref within +/- max_shift."""

    corr = np.correlate(sig, ref, mode="full")
    shift = -int(np.argmax(corr) - (len(ref) - 1))
    return int(np.clip(shift, -max_shift, max_shift))


def _shift_array(sig: np.ndarray, shift: int) -> np.ndarray:
    """Shift array using circular roll (approximate warp)."""

    return np.roll(sig, shift)


# ----------------------------- Step base -----------------------------


@dataclass
class Step:
    """Base preprocessing step."""

    name: str
    config: Dict[str, Any]

    def fit(self, ds: FoodSpectrumSet) -> "Step":
        return self

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, **self.config}

    def hash(self) -> str:
        return _short_hash(self.to_dict())


# ----------------------------- Baseline -----------------------------


def _baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    # Simple ALS baseline using finite differences
    L = len(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = W + lam * (D.T @ D)
        z = np.linalg.solve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _baseline_poly(y: np.ndarray, degree: int = 3) -> np.ndarray:
    x = np.linspace(0, 1, y.shape[0])
    coef = np.polyfit(x, y, degree)
    return np.polyval(coef, x)


def _baseline_snip(y: np.ndarray, iterations: int = 30, window: int = 5) -> np.ndarray:
    # Simplified SNIP: iterative clipping of log spectrum
    s = np.log(np.clip(y - y.min() + 1e-6, 1e-6, None))
    s_work = s.copy()
    for k in range(1, iterations + 1):
        s_shift = (np.roll(s_work, k) + np.roll(s_work, -k)) / 2
        s_work = np.minimum(s_work, s_shift)
    baseline = np.exp(s_work) + y.min()
    return baseline


def _baseline_rubberband(y: np.ndarray) -> np.ndarray:
    x = np.arange(y.shape[0])
    order = np.argsort(x)
    lower = []

    def cross(o, a, b):
        return (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o])

    for idx in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], idx) <= 0:
            lower.pop()
        lower.append(idx)
    baseline = np.interp(x, x[lower], y[lower])
    return baseline


class BaselineStep(Step):
    def __init__(self, method: str = "als", **kwargs: Any):
        super().__init__(name="baseline", config={"method": method, **kwargs})

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        X = _ensure_2d(ds.x)
        method = self.config.get("method", "als")
        params = self.config
        baselines = np.zeros_like(X)
        for i, row in enumerate(X):
            if method == "als":
                baselines[i] = _baseline_als(row, lam=params.get("lam", 1e5), p=params.get("p", 0.01))
            elif method == "airpls":
                baselines[i] = _baseline_als(row, lam=params.get("lam", 1e5), p=params.get("p", 0.001))
            elif method == "snip":
                baselines[i] = _baseline_snip(row, iterations=params.get("iterations", 30))
            elif method == "poly":
                baselines[i] = _baseline_poly(row, degree=params.get("degree", 3))
            elif method == "rubberband":
                baselines[i] = _baseline_rubberband(row)
            else:
                baselines[i] = 0.0
        corrected = X - baselines
        return _clone_ds(ds, corrected)


# ----------------------------- Smoothing -----------------------------


class SmoothingStep(Step):
    def __init__(self, method: str = "savgol", **kwargs: Any):
        super().__init__(name="smoothing", config={"method": method, **kwargs})

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        X = _ensure_2d(ds.x)
        method = self.config.get("method", "savgol")
        params = self.config
        if method == "savgol":
            wl = params.get("window_length", 7)
            po = params.get("polyorder", 3)
            X_s = savgol_filter(X, window_length=wl, polyorder=po, axis=1, mode="interp")
        elif method == "moving_average":
            w = params.get("window", 5)
            kernel = np.ones(w) / w
            X_s = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, X)
        elif method == "gaussian":
            sigma = params.get("sigma", 1.0)
            X_s = gaussian_filter1d(X, sigma=sigma, axis=1)
        else:
            X_s = X
        return _clone_ds(ds, X_s)


# ----------------------------- Normalization -----------------------------


def _norm_vector(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


def _norm_area(X: np.ndarray) -> np.ndarray:
    denom = np.sum(X, axis=1, keepdims=True) + 1e-12
    return X / denom


def _norm_max(X: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(X), axis=1, keepdims=True) + 1e-12
    return X / denom


def _norm_snv(X: np.ndarray) -> np.ndarray:
    mu = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-12
    return (X - mu) / std


def _norm_msc(X: np.ndarray) -> np.ndarray:
    ref = np.mean(X, axis=0, keepdims=True)
    out = np.zeros_like(X)
    for i, row in enumerate(X):
        b = np.linalg.lstsq(ref.T, row, rcond=None)[0]
        slope = b[0] if b.size == 1 else b.mean()
        intercept = 0.0
        out[i] = (row - intercept) / (slope + 1e-12)
    return out


class NormalizationStep(Step):
    def __init__(self, method: str = "vector", **kwargs: Any):
        super().__init__(name="normalization", config={"method": method, **kwargs})

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        X = _ensure_2d(ds.x)
        method = self.config.get("method", "vector")
        if method == "vector":
            Xn = _norm_vector(X)
        elif method == "area":
            Xn = _norm_area(X)
        elif method == "max":
            Xn = _norm_max(X)
        elif method == "snv":
            Xn = _norm_snv(X)
        elif method == "msc":
            Xn = _norm_msc(X)
        else:
            Xn = X
        return _clone_ds(ds, Xn)


# ----------------------------- Derivatives -----------------------------


class DerivativeStep(Step):
    def __init__(self, order: int = 1, window_length: int = 7, polyorder: int = 2):
        super().__init__(name="derivative", config={"order": order, "window_length": window_length, "polyorder": polyorder})

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        X = _ensure_2d(ds.x)
        order = self.config["order"]
        wl = self.config["window_length"]
        po = self.config["polyorder"]
        Xd = savgol_filter(X, window_length=wl, polyorder=po, deriv=order, axis=1, mode="interp")
        return _clone_ds(ds, Xd)


# ----------------------------- Alignment (v2 placeholders) -----------------------------


class AlignmentStep(Step):
    def __init__(self, method: str = "cow", **kwargs: Any):
        super().__init__(name="alignment", config={"method": method, **kwargs})

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        method = self.config.get("method", "cow")
        max_shift = int(self.config.get("max_shift", 10))
        ref_strategy = self.config.get("reference", "median")

        X = _ensure_2d(ds.x)
        if ref_strategy == "first":
            ref = X[0]
        else:
            ref = np.median(X, axis=0)

        shifts = []
        aligned = np.zeros_like(X)
        for i, row in enumerate(X):
            if method == "none":
                shift = 0
            elif method == "peak":
                ref_peak = int(np.argmax(ref))
                row_peak = int(np.argmax(row))
                shift = int(np.clip(ref_peak - row_peak, -max_shift, max_shift))
            else:  # default to cow-lite via cross-correlation shift
                shift = _xcorr_shift(ref, row, max_shift=max_shift)
            shifts.append(shift)
            aligned[i] = _shift_array(row, shift)

        # Stash shifts for downstream metrics
        self.config["_last_shifts"] = shifts
        self.config["_last_ref"] = ref.tolist()
        return _clone_ds(ds, aligned)


# ----------------------------- Resampling / grid unification -----------------------------


class ResampleStep(Step):
    def __init__(self, grid: Optional[np.ndarray] = None, method: str = "linear"):
        super().__init__(name="resample", config={"method": method, "grid": grid.tolist() if isinstance(grid, np.ndarray) else grid})
        self.grid = grid

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        if self.grid is None:
            return ds
        X = _ensure_2d(ds.x)
        source_grid = ds.wavenumbers
        target = np.asarray(self.grid, dtype=float)
        if self.config.get("method", "linear") == "cubic":
            # Fallback to linear for simplicity if scipy CubicSpline unavailable
            interp_fn = lambda row: np.interp(target, source_grid, row)
        else:
            interp_fn = lambda row: np.interp(target, source_grid, row)
        Xr = np.vstack([interp_fn(row) for row in X])
        return _clone_ds(ds, Xr, wavenumbers=target)


# ----------------------------- Metrics -----------------------------


def baseline_metrics(X: np.ndarray) -> Dict[str, float]:
    X = _ensure_2d(X)
    # Low-frequency energy via FFT first 5% of spectrum
    fft = np.abs(np.fft.rfft(X, axis=1))
    cutoff = max(1, int(0.05 * fft.shape[1]))
    lowfreq = np.mean(fft[:, :cutoff])
    negativity = float(np.mean(X < 0))
    # Smoothness: second-derivative energy
    d2 = np.diff(X, n=2, axis=1)
    smooth_pen = float(np.mean(d2**2))
    return {
        "baseline_residual_energy_lowfreq": float(lowfreq),
        "negativity_fraction": negativity,
        "smoothness_penalty": smooth_pen,
    }


def smoothing_metrics(raw: np.ndarray, smoothed: np.ndarray) -> Dict[str, float]:
    raw = _ensure_2d(raw)
    sm = _ensure_2d(smoothed)
    # SNR gain (variance ratio)
    snr_gain = float(np.mean(np.var(raw, axis=1) / (np.var(sm, axis=1) + 1e-9)))
    # Peak preservation: max absolute diff percent
    peak_pres = float(np.mean(np.max(np.abs(raw - sm), axis=1) / (np.max(np.abs(raw), axis=1) + 1e-9)))
    return {"snr_gain": snr_gain, "peak_preservation": peak_pres}


def normalization_metrics(X: np.ndarray) -> Dict[str, float]:
    X = _ensure_2d(X)
    norms = np.linalg.norm(X, axis=1)
    cv = float(np.std(norms) / (np.mean(norms) + 1e-9))
    return {"scale_stability": cv}


def alignment_metrics(before: np.ndarray, after: np.ndarray, ref: np.ndarray, shifts: List[int]) -> Dict[str, float]:
    before = _ensure_2d(before)
    after = _ensure_2d(after)
    ref = np.asarray(ref, dtype=float)
    before_err = np.mean((before - ref) ** 2)
    after_err = np.mean((after - ref) ** 2)
    improvement = float(before_err - after_err)
    mean_abs_shift = float(np.mean(np.abs(shifts))) if shifts else 0.0
    max_abs_shift = float(np.max(np.abs(shifts))) if shifts else 0.0
    return {
        "alignment_residual_energy": float(after_err),
        "alignment_improvement": improvement,
        "mean_abs_shift": mean_abs_shift,
        "max_abs_shift": max_abs_shift,
    }


# ----------------------------- Pipeline -----------------------------


@dataclass
class PreprocessPipeline:
    steps: List[Step] = field(default_factory=list)

    def add(self, step: Step) -> "PreprocessPipeline":
        self.steps.append(step)
        return self

    def fit(self, ds: FoodSpectrumSet) -> "PreprocessPipeline":
        for step in self.steps:
            step.fit(ds)
        return self

    def transform(self, ds: FoodSpectrumSet) -> Tuple[FoodSpectrumSet, Dict[str, Dict[str, float]]]:
        metrics: Dict[str, Dict[str, float]] = {}
        current = ds
        raw_before = ds.x
        for step in self.steps:
            before = current.x
            current = step.transform(current)
            after = current.x
            if step.name == "baseline":
                metrics["baseline"] = baseline_metrics(after)
            elif step.name == "smoothing":
                metrics["smoothing"] = smoothing_metrics(before, after)
            elif step.name == "normalization":
                metrics["normalization"] = normalization_metrics(after)
            elif step.name == "alignment":
                # Use pre-step reference (median of before) for alignment quality
                ref = np.asarray(step.config.get("_last_ref", np.median(before, axis=0)))
                shifts = step.config.get("_last_shifts", [])
                metrics["alignment"] = alignment_metrics(before, after, ref=ref, shifts=shifts)
        return current, metrics

    def to_dict(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.steps]

    def hash(self) -> str:
        return _short_hash({"steps": self.to_dict()})


# ----------------------------- AutoPreprocess -----------------------------


def _score_metrics(m: Dict[str, Dict[str, float]]) -> float:
    # Lower is better for these metrics; invert where needed
    baseline = m.get("baseline", {})
    smoothing = m.get("smoothing", {})
    norm = m.get("normalization", {})
    align = m.get("alignment", {})

    score = 0.0
    score -= baseline.get("negativity_fraction", 0.0) * 2.0
    score -= baseline.get("smoothness_penalty", 0.0) * 0.1
    score += smoothing.get("snr_gain", 0.0) * 0.5
    score -= smoothing.get("peak_preservation", 0.0) * 0.5
    score -= norm.get("scale_stability", 0.0) * 0.2
    score += align.get("alignment_improvement", 0.0) * 0.5
    score -= align.get("mean_abs_shift", 0.0) * 0.05
    score -= align.get("alignment_residual_energy", 0.0) * 0.01
    return score


@dataclass
class AutoPreprocessResult:
    pipeline: PreprocessPipeline
    metrics: Dict[str, Dict[str, float]]
    score: float
    explanation: str


class AutoPreprocess:
    """Search over candidate pipelines and select best by heuristic score."""

    def __init__(
        self,
        baselines: Sequence[Dict[str, Any]] | None = None,
        smoothers: Sequence[Dict[str, Any]] | None = None,
        aligners: Sequence[Dict[str, Any]] | None = None,
        normalizers: Sequence[Dict[str, Any]] | None = None,
        derivatives: Sequence[Dict[str, Any]] | None = None,
    ) -> None:
        self.baselines = baselines or [
            {"method": "als", "lam": 1e5, "p": 0.01},
            {"method": "rubberband"},
        ]
        self.smoothers = smoothers or [
            {"method": "savgol", "window_length": 7, "polyorder": 3},
            {"method": "moving_average", "window": 5},
        ]
        self.aligners = aligners or [
            {"method": "none"},
            {"method": "cow", "max_shift": 8},
            {"method": "peak", "max_shift": 5},
        ]
        self.normalizers = normalizers or [
            {"method": "snv"},
            {"method": "vector"},
            {"method": "area"},
        ]
        self.derivatives = derivatives or [
            {"order": 0},
            {"order": 1, "window_length": 9, "polyorder": 2},
        ]

    def _build_pipeline(self, b_cfg: Dict[str, Any], s_cfg: Dict[str, Any], a_cfg: Dict[str, Any], n_cfg: Dict[str, Any], d_cfg: Dict[str, Any]) -> PreprocessPipeline:
        steps: List[Step] = []
        if b_cfg.get("method", "none") != "none":
            steps.append(BaselineStep(**b_cfg))
        if s_cfg.get("method", "none") != "none":
            steps.append(SmoothingStep(**s_cfg))
        if a_cfg.get("method", "none") != "none":
            steps.append(AlignmentStep(**a_cfg))
        if n_cfg.get("method", "none") != "none":
            steps.append(NormalizationStep(**n_cfg))
        if d_cfg.get("order", 0) > 0:
            steps.append(DerivativeStep(**d_cfg))
        return PreprocessPipeline(steps)

    def search(self, ds: FoodSpectrumSet, max_candidates: int = 20) -> AutoPreprocessResult:
        combos = list(product(self.baselines, self.smoothers, self.aligners, self.normalizers, self.derivatives))
        combos = combos[:max_candidates]

        best_score = -math.inf
        best_result: Optional[AutoPreprocessResult] = None

        # Use a subset of data for scoring (first 32 samples)
        subset = ds
        if ds.x.shape[0] > 32:
            meta_subset = None
            if isinstance(ds.metadata, pd.DataFrame):
                meta_subset = ds.metadata.iloc[:32].reset_index(drop=True)
            subset = ds.__class__(
                x=ds.x[:32],
                wavenumbers=ds.wavenumbers,
                metadata=meta_subset,
                modality=ds.modality,
            )

        for b_cfg, s_cfg, a_cfg, n_cfg, d_cfg in combos:
            pipeline = self._build_pipeline(b_cfg, s_cfg, a_cfg, n_cfg, d_cfg)
            transformed, metrics = pipeline.transform(subset)
            score = _score_metrics(metrics)
            if score > best_score:
                best_score = score
                explanation = (
                    f"Selected baseline={b_cfg} smoothing={s_cfg} alignment={a_cfg} normalization={n_cfg} derivative={d_cfg} "
                    f"with score={score:.4f}"
                )
                best_result = AutoPreprocessResult(pipeline, metrics, score, explanation)

        if best_result is None:
            raise RuntimeError("AutoPreprocess could not evaluate any pipeline")
        return best_result


__all__ = [
    "Step",
    "BaselineStep",
    "SmoothingStep",
    "NormalizationStep",
    "DerivativeStep",
    "AlignmentStep",
    "ResampleStep",
    "PreprocessPipeline",
    "AutoPreprocess",
    "AutoPreprocessResult",
    "baseline_metrics",
    "smoothing_metrics",
    "normalization_metrics",
    "alignment_metrics",
]
