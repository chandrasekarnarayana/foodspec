"""Gage R&R (Repeatability & Reproducibility) Analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

__all__ = ["GageRR", "MeasurementSystemAnalysis"]


class GageRR:
    """
    Gauge Repeatability and Reproducibility (Gage R&R) Analysis.

    Decomposes measurement variation into:
    - Repeatability (equipment/within-appraiser variation)
    - Reproducibility (between-appraiser variation)
    - Part-to-part variation
    - Total variation
    """

    def __init__(self, confidence: float = 0.95):
        """
        Initialize Gage R&R analyzer.

        Parameters
        ----------
        confidence : float, default=0.95
            Confidence level for analysis (typically 0.95 or 0.99).

        Notes
        -----
        Standard Gage R&R format:
        - Crossed design: Each operator (appraiser) measures each part
        - Multiple replicate measurements per part-operator combination

        Assumes balanced design:
        n_parts × n_operators × n_replicates measurements
        """
        self.confidence = confidence
        self.results_ = None

    def analyze_crossed(
        self,
        measurements: np.ndarray,
        parts: np.ndarray,
        operators: np.ndarray,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform Gage R&R analysis for crossed design.

        Parameters
        ----------
        measurements : np.ndarray, shape (n_measurements,)
            All measurements.
        parts : np.ndarray, shape (n_measurements,)
            Part identifiers (1..n_parts).
        operators : np.ndarray, shape (n_measurements,)
            Operator/appraiser identifiers (1..n_operators).
        tolerance : float, optional
            Tolerance (USL - LSL). If None, uses 6*sigma_part_to_part.

        Returns
        -------
        results : dict
            Gage R&R analysis results.

        Notes
        -----
        Model: X_ijr = μ + p_i + o_j + (po)_ij + e_ijr
        where:
        - p_i: Part effect
        - o_j: Operator (reproducibility) effect
        - (po)_ij: Part-operator interaction
        - e_ijr: Repeatability (measurement error)
        """
        measurements = np.asarray(measurements, dtype=np.float64).ravel()
        parts = np.asarray(parts, dtype=np.int32).ravel()
        operators = np.asarray(operators, dtype=np.int32).ravel()

        if not (len(measurements) == len(parts) == len(operators)):
            raise ValueError("measurements, parts, and operators must have same length")

        n_parts = len(np.unique(parts))
        n_operators = len(np.unique(operators))
        n_measurements = len(measurements)
        n_replicates = n_measurements // (n_parts * n_operators)

        if n_measurements != n_parts * n_operators * n_replicates:
            raise ValueError(f"Unbalanced design: {n_measurements} != {n_parts}*{n_operators}*{n_replicates}")

        # Grand mean
        grand_mean = np.mean(measurements)

        # --- Variance Components ---

        # Within-group (repeatability): variation within each part-operator combination
        repeatability_ss = 0
        for p in np.unique(parts):
            for o in np.unique(operators):
                mask = (parts == p) & (operators == o)
                group_measurements = measurements[mask]
                if len(group_measurements) > 1:
                    repeatability_ss += np.sum((group_measurements - np.mean(group_measurements)) ** 2)

        df_repeatability = n_parts * n_operators * (n_replicates - 1)
        ms_repeatability = repeatability_ss / df_repeatability if df_repeatability > 0 else 0
        var_repeatability = ms_repeatability

        # Part-to-part variation
        part_means = np.array([np.mean(measurements[parts == p]) for p in np.unique(parts)])
        part_ss = n_operators * n_replicates * np.sum((part_means - grand_mean) ** 2)
        df_part = n_parts - 1
        ms_part = part_ss / df_part if df_part > 0 else 0
        var_part = (ms_part - ms_repeatability) / (n_operators * n_replicates) if ms_part > ms_repeatability else 0

        # Operator (reproducibility)
        operator_means = np.array([np.mean(measurements[operators == o]) for o in np.unique(operators)])
        operator_ss = n_parts * n_replicates * np.sum((operator_means - grand_mean) ** 2)
        df_operator = n_operators - 1
        ms_operator = operator_ss / df_operator if df_operator > 0 else 0
        var_operator = (
            (ms_operator - ms_repeatability) / (n_parts * n_replicates) if ms_operator > ms_repeatability else 0
        )

        # Part-operator interaction
        interaction_ss = 0
        for p in np.unique(parts):
            for o in np.unique(operators):
                mask = (parts == p) & (operators == o)
                group_mean = np.mean(measurements[mask])
                part_mean_p = np.mean(measurements[parts == p])
                operator_mean_o = np.mean(measurements[operators == o])
                effect = group_mean - part_mean_p - operator_mean_o + grand_mean
                interaction_ss += n_replicates * (effect**2)

        df_interaction = (n_parts - 1) * (n_operators - 1)
        ms_interaction = interaction_ss / df_interaction if df_interaction > 0 else 0
        var_interaction = (ms_interaction - ms_repeatability) / n_replicates if ms_interaction > ms_repeatability else 0

        # Total variation
        var_total = var_part + var_operator + var_repeatability + var_interaction

        # Reproducibility = Operator + Interaction
        var_reproducibility = var_operator + var_interaction

        # Gage R&R = Repeatability + Reproducibility
        var_gage_rr = var_repeatability + var_reproducibility

        # Use tolerance if provided, else 6*sigma_part
        if tolerance is None:
            tolerance = 6 * np.sqrt(max(var_part, 0))

        if tolerance == 0:
            tolerance = 6 * np.std(measurements, ddof=1)

        # Calculate standard deviations
        std_repeatability = np.sqrt(var_repeatability)
        std_reproducibility = np.sqrt(var_reproducibility)
        std_gage_rr = np.sqrt(var_gage_rr)
        std_part_to_part = np.sqrt(var_part)
        std_total = np.sqrt(var_total)

        # Calculate percentage of tolerance
        pct_repeatability = (6 * std_repeatability / tolerance) * 100 if tolerance > 0 else 0
        pct_reproducibility = (6 * std_reproducibility / tolerance) * 100 if tolerance > 0 else 0
        pct_gage_rr = (6 * std_gage_rr / tolerance) * 100 if tolerance > 0 else 0

        # Number of distinct categories (NDC)
        # Should be ≥ 5 for acceptable measurement system
        ndc = 1.41 * std_part_to_part / std_gage_rr if std_gage_rr > 0 and var_part > 0 else 0
        ndc = max(0, ndc)

        self.results_ = {
            "n_parts": n_parts,
            "n_operators": n_operators,
            "n_replicates": n_replicates,
            "n_measurements": n_measurements,
            "grand_mean": float(grand_mean),
            "variance_components": {
                "part_to_part": float(var_part),
                "operator": float(var_operator),
                "part_operator_interaction": float(var_interaction),
                "repeatability": float(var_repeatability),
                "reproducibility": float(var_reproducibility),
                "gage_rr": float(var_gage_rr),
                "total": float(var_total),
            },
            "std_components": {
                "part_to_part": float(std_part_to_part),
                "operator": float(np.sqrt(var_operator)),
                "interaction": float(np.sqrt(var_interaction)),
                "repeatability": float(std_repeatability),
                "reproducibility": float(std_reproducibility),
                "gage_rr": float(std_gage_rr),
                "total": float(std_total),
            },
            "tolerance": float(tolerance),
            "percent_tolerance": {
                "repeatability": float(pct_repeatability),
                "reproducibility": float(pct_reproducibility),
                "gage_rr": float(pct_gage_rr),
                "part_to_part": float((6 * std_part_to_part / tolerance) * 100) if tolerance > 0 else 0,
            },
            "ndc": float(ndc),
            "acceptability": self._classify_acceptability(pct_gage_rr, ndc),
        }

        return self.results_

    @staticmethod
    def _classify_acceptability(pct_gage_rr: float, ndc: float) -> Dict[str, str]:
        """
        Classify measurement system acceptability.

        Parameters
        ----------
        pct_gage_rr : float
            Gage R&R as percentage of tolerance.
        ndc : float
            Number of distinct categories.

        Returns
        -------
        classification : dict
            Acceptability ratings.
        """
        if pct_gage_rr <= 10:
            gage_rr_rating = "Acceptable"
        elif pct_gage_rr <= 30:
            gage_rr_rating = "Marginal"
        else:
            gage_rr_rating = "Unacceptable"

        if ndc >= 5:
            ndc_rating = "Acceptable"
        elif ndc >= 3:
            ndc_rating = "Marginal"
        else:
            ndc_rating = "Unacceptable"

        return {
            "gage_rr": gage_rr_rating,
            "ndc": ndc_rating,
        }

    def report(self) -> str:
        """
        Generate Gage R&R analysis report.

        Returns
        -------
        report : str
            Formatted report.
        """
        if self.results_ is None:
            return "No analysis performed. Call analyze_crossed() first."

        r = self.results_
        v = r["variance_components"]
        s = r["std_components"]
        p = r["percent_tolerance"]

        report = f"""
Gage R&R Analysis Report
========================
Design:                     Crossed
Parts:                      {r["n_parts"]}
Operators:                  {r["n_operators"]}
Replicates:                 {r["n_replicates"]}
Total Measurements:         {r["n_measurements"]}
Grand Mean:                 {r["grand_mean"]:.6f}
Tolerance:                  {r["tolerance"]:.6f}

Variance Components:
  Part-to-Part:             {v["part_to_part"]:.6e}
  Repeatability (Equip):    {v["repeatability"]:.6e}
  Operator (Appraisers):    {v["operator"]:.6e}
  Part×Operator Inter:      {v["part_operator_interaction"]:.6e}
  Reproducibility:          {v["reproducibility"]:.6e}
  Gage R&R:                 {v["gage_rr"]:.6e}
  Total:                    {v["total"]:.6e}

Standard Deviations:
  Part-to-Part:             {s["part_to_part"]:.6f}
  Repeatability:            {s["repeatability"]:.6f}
  Reproducibility:          {s["reproducibility"]:.6f}
  Gage R&R:                 {s["gage_rr"]:.6f}
  Total:                    {s["total"]:.6f}

%Tolerance Analysis:
  Repeatability:            {p["repeatability"]:.2f}%
  Reproducibility:          {p["reproducibility"]:.2f}%
  Gage R&R:                 {p["gage_rr"]:.2f}%
  Part-to-Part:             {p["part_to_part"]:.2f}%

Number of Distinct Categories (NDC): {r["ndc"]:.1f}

Acceptability:
  Gage R&R:                 {r["acceptability"]["gage_rr"]}
  NDC:                      {r["acceptability"]["ndc"]}

Recommendations:
"""
        if r["acceptability"]["gage_rr"] == "Unacceptable":
            report += "  - Measurement system needs improvement\n"
            report += "  - Investigate major sources of variation\n"
        elif r["acceptability"]["gage_rr"] == "Marginal":
            report += "  - Consider measurement system improvements\n"

        if r["acceptability"]["ndc"] != "Acceptable":
            report += "  - System cannot distinguish enough part categories\n"

        return report


class MeasurementSystemAnalysis:
    """Comprehensive Measurement System Analysis combining Gage R&R with additional metrics."""

    def __init__(self, tolerance: Optional[float] = None, confidence: float = 0.95):
        """
        Initialize measurement system analyzer.

        Parameters
        ----------
        tolerance : float, optional
            Measurement tolerance.
        confidence : float, default=0.95
            Confidence level.
        """
        self.tolerance = tolerance
        self.confidence = confidence
        self.gage_rr = GageRR(confidence=confidence)

    def analyze(
        self,
        measurements: np.ndarray,
        parts: np.ndarray,
        operators: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive measurement system analysis.

        Parameters
        ----------
        measurements : np.ndarray
            All measurements.
        parts : np.ndarray
            Part identifiers.
        operators : np.ndarray
            Operator identifiers.

        Returns
        -------
        results : dict
            Complete analysis results.
        """
        return self.gage_rr.analyze_crossed(measurements, parts, operators, tolerance=self.tolerance)

    def report(self) -> str:
        """Get formatted analysis report."""
        return self.gage_rr.report()
