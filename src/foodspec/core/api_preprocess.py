"""Preprocessing mixin for FoodSpec API - QC and preprocessing."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

import numpy as np

from foodspec.preprocess.engine import AutoPreprocess
from foodspec.qc.engine import generate_qc_report


class FoodSpecPreprocessMixin:
    """Mixin class providing preprocessing capabilities for FoodSpec."""

    def qc(
        self,
        method: str = "robust_z",
        threshold: float = 0.5,
        **kwargs,
    ):
        """Apply QC (quality control) to detect and flag outliers.

        Parameters
        ----------
        method : {'robust_z', 'mahalanobis', 'isolation_forest', 'lof'}, optional
            Outlier detection method. Default: 'robust_z'.
        threshold : float, optional
            Not used in current implementation (reserved for future scoring cutoffs).
        **kwargs
            Additional arguments forwarded to QC engine (e.g., reference_grid, batch_col, time_col).

        Returns
        -------
        FoodSpec
            Self (for chaining).
        """

        report = generate_qc_report(
            self.data,
            reference_grid=kwargs.get("reference_grid"),
            batch_col=kwargs.get("batch_col"),
            time_col=kwargs.get("time_col"),
            outlier_method=method,
        )

        # Record metrics and diagnostics
        self.bundle.add_metrics("qc_health", report.health.aggregates)
        self.bundle.add_metrics("qc_outliers", {"outlier_rate": float(report.outliers.labels.mean())})
        self.bundle.add_metrics("qc_drift", {"drift_score": report.drift.drift_score, "trend_slope": report.drift.trend_slope})
        self.bundle.add_diagnostic("qc_health_table", report.health.table.to_dict(orient="list"))
        self.bundle.add_diagnostic("qc_outlier_scores", report.outliers.scores.tolist())
        self.bundle.add_diagnostic("qc_recommendation", report.recommendations)

        step_hash = hashlib.sha256(json.dumps({"method": method, "recommendation": report.recommendations}, sort_keys=True).encode()).hexdigest()[:8]
        self.bundle.run_record.add_step(
            "qc",
            step_hash,
            metadata={"method": method, "recommendation": report.recommendations},
        )

        self._steps_applied.append("qc")
        return self

    def preprocess(
        self,
        preset: str = "auto",
        **kwargs,
    ):
        """Apply preprocessing pipeline.

        Parameters
        ----------
        preset : str, optional
            Preset name: 'auto', 'quick', 'standard', 'publication'. Default: 'auto'.
        **kwargs
            Override preset parameters (forwarded to AutoPreprocess or manual presets).

        Returns
        -------
        FoodSpec
            Self (for chaining).
        """


        # Use AutoPreprocess as the default preset (Phase 3).
        if preset == "auto":
            auto = AutoPreprocess(
                baselines=kwargs.get("baselines"),
                smoothers=kwargs.get("smoothers"),
                normalizers=kwargs.get("normalizers"),
                derivatives=kwargs.get("derivatives"),
            )
            result = auto.search(self.data)
            processed, metrics = result.pipeline.transform(self.data)
            self.data = processed

            # Record metrics and explanation
            self.bundle.add_metrics("preprocess", metrics)
            self.bundle.add_diagnostic("preprocess", {"explanation": result.explanation})

            step_hash = result.pipeline.hash()
            self.bundle.run_record.add_step(
                "preprocess",
                step_hash,
                metadata={"preset": preset, "pipeline": result.pipeline.to_dict(), "metrics": metrics},
            )
        else:
            # Simple fallback presets (quick/standard/publication) using AutoPreprocess with narrow grids
            preset_map = {
                "quick": {
                    "baselines": [{"method": "rubberband"}],
                    "smoothers": [{"method": "moving_average", "window": 3}],
                    "normalizers": [{"method": "snv"}],
                    "derivatives": [{"order": 0}],
                },
                "standard": {
                    "baselines": [{"method": "als", "lam": 1e5, "p": 0.01}],
                    "smoothers": [{"method": "savgol", "window_length": 7, "polyorder": 3}],
                    "normalizers": [{"method": "vector"}],
                    "derivatives": [{"order": 1, "window_length": 9, "polyorder": 2}],
                },
                "publication": {
                    "baselines": [{"method": "als", "lam": 1e6, "p": 0.001}],
                    "smoothers": [{"method": "savgol", "window_length": 11, "polyorder": 3}],
                    "normalizers": [{"method": "msc"}],
                    "derivatives": [{"order": 1, "window_length": 15, "polyorder": 3}],
                },
            }
            preset_cfg = preset_map.get(preset)
            if preset_cfg is None:
                raise ValueError(f"Unknown preprocessing preset: {preset}")
            auto = AutoPreprocess(**preset_cfg)
            result = auto.search(self.data)
            processed, metrics = result.pipeline.transform(self.data)
            self.data = processed
            self.bundle.add_metrics("preprocess", metrics)
            self.bundle.add_diagnostic("preprocess", {"explanation": result.explanation})
            step_hash = result.pipeline.hash()
            self.bundle.run_record.add_step(
                "preprocess",
                step_hash,
                metadata={"preset": preset, "pipeline": result.pipeline.to_dict(), "metrics": metrics},
            )

        self._steps_applied.append(f"preprocess({preset})")
        return self

    def apply_matrix_correction(
        self,
        method: Literal["background_air", "background_dark", "adaptive_baseline", "none"] = "adaptive_baseline",
        scaling: Literal["median_mad", "huber", "mcd", "none"] = "median_mad",
        domain_adapt: bool = False,
        matrix_column: Optional[str] = None,
        reference_spectra: Optional[np.ndarray] = None,
    ):
        """
        Apply matrix correction to remove matrix effects (e.g., chips vs. pure oil).

        **Key Assumptions:**
        - Background reference spectra measured under identical conditions
        - Matrix types known/inferrable from metadata
        - Domain adaptation requires ≥2 matrix types with ≥10 samples each
        - Spectral ranges aligned before correction

        See foodspec.matrix_correction module docstring for full details.

        Parameters
        ----------
        method : str, default='adaptive_baseline'
            Background subtraction method.
        scaling : str, default='median_mad'
            Robust scaling method per matrix type.
        domain_adapt : bool, default=False
            Whether to apply subspace alignment between matrices.
        matrix_column : str, optional
            Metadata column with matrix type labels.
        reference_spectra : np.ndarray, optional
            Background reference (for background_air/dark methods).

        Returns
        -------
        FoodSpec
            Self (for chaining).
        """
        from foodspec.preprocess.matrix_correction import apply_matrix_correction as _apply_mc

        self.data, mc_metrics = _apply_mc(
            self.data,
            method=method,
            scaling=scaling,
            domain_adapt=domain_adapt,
            matrix_column=matrix_column,
            reference_spectra=reference_spectra,
        )

        # Record metrics
        for key, val in mc_metrics.items():
            self.bundle.add_metrics(f"matrix_correction_{key}", val)

        self.bundle.run_record.add_step(
            "matrix_correction",
            hashlib.sha256(json.dumps(mc_metrics, sort_keys=True).encode()).hexdigest()[:8],
            metadata={"method": method, "scaling": scaling, "domain_adapt": domain_adapt},
        )
        self._steps_applied.append("matrix_correction")

        return self

    def apply_calibration_transfer(
        self,
        source_standards: np.ndarray,
        target_standards: np.ndarray,
        method: Literal["ds", "pds"] = "ds",
        pds_window_size: int = 11,
        alpha: float = 1.0,
    ):
        """
        Apply calibration transfer to align target instrument to source.

        **Key Assumptions:**
        - Source/target standards are paired (same samples measured on both)
        - Standards span the calibration range
        - Spectral alignment already performed
        - Linear transformation adequate for instrument differences

        See foodspec.calibration_transfer module docstring for full details.

        Parameters
        ----------
        source_standards : np.ndarray, shape (n_standards, n_wavenumbers)
            Source (reference) instrument spectra.
        target_standards : np.ndarray, shape (n_standards, n_wavenumbers)
            Target (slave) instrument spectra.
        method : {'ds', 'pds'}, default='ds'
            Transfer method (Direct Standardization or Piecewise DS).
        pds_window_size : int, default=11
            PDS window size (ignored if method='ds').
        alpha : float, default=1.0
            Ridge regularization parameter.

        Returns
        -------
        FoodSpec
            Self (for chaining).
        """
        from foodspec.preprocess.calibration_transfer import calibration_transfer_workflow

        self.data.x, ct_metrics = calibration_transfer_workflow(
            source_standards,
            target_standards,
            self.data.x,
            method=method,
            pds_window_size=pds_window_size,
            alpha=alpha,
        )

        # Record metrics
        for key, val in ct_metrics.items():
            self.bundle.add_metrics(f"calibration_transfer_{key}", val)
        # Generate simple HTML dashboard if metrics include success metrics
        try:
            from foodspec.calibration_transfer_dashboard import build_dashboard_html
            html = build_dashboard_html(ct_metrics)
            self.bundle.add_diagnostic("calibration_transfer_dashboard", html)
        except Exception:
            pass

        self.bundle.run_record.add_step(
            "calibration_transfer",
            hashlib.sha256(json.dumps(ct_metrics, sort_keys=True).encode()).hexdigest()[:8],
            metadata={"method": method, "pds_window_size": pds_window_size},
        )
        self._steps_applied.append("calibration_transfer")

        return self
