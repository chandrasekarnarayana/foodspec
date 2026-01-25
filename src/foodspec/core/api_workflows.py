"""Workflows mixin for FoodSpec API - heating trajectory analysis."""
from __future__ import annotations


import hashlib
import json
from typing import Any, Dict, List, Optional


class FoodSpecWorkflowsMixin:
    """Mixin class providing workflow analysis capabilities for FoodSpec."""

    def analyze_heating_trajectory(
        self,
        time_column: str,
        indices: List[str] = ["pi", "tfc", "oit_proxy"],
        classify_stages: bool = False,
        stage_column: Optional[str] = None,
        estimate_shelf_life: bool = False,
        shelf_life_threshold: Optional[float] = None,
        shelf_life_index: str = "pi",
    ) -> Dict[str, Any]:
        """
        Analyze heating/oxidation trajectory from time-series spectra.

        **Key Assumptions:**
        - time_column exists and is numeric (hours, days, timestamps)
        - Repeated measurements over time (longitudinal data)
        - Degradation is monotonic or follows known patterns
        - ≥5 time points per sample/group for reliable regression
        - No major batch effects confounding time trends

        See foodspec.workflows.heating_trajectory module docstring for full details.

        Parameters
        ----------
        time_column : str
            Metadata column with time values.
        indices : list of str, default=['pi', 'tfc', 'oit_proxy']
            Oxidation indices to extract and model.
        classify_stages : bool, default=False
            Whether to train degradation stage classifier.
        stage_column : str, optional
            Metadata column with stage labels (required if classify_stages=True).
        estimate_shelf_life : bool, default=False
            Whether to estimate shelf life.
        shelf_life_threshold : float, optional
            Threshold for shelf-life criterion (required if estimate_shelf_life=True).
        shelf_life_index : str, default='pi'
            Index to use for shelf-life estimation.

        Returns
        -------
        results : dict
            - 'indices': extracted indices DataFrame
            - 'trajectory_models': fit metrics per index
            - 'stage_classification' (if enabled): classification metrics
            - 'shelf_life' (if enabled): shelf-life estimation
        """
        from foodspec.workflows.heating_trajectory import analyze_heating_trajectory as _analyze_ht

        results = _analyze_ht(
            self.data,
            time_column=time_column,
            indices=indices,
            classify_stages=classify_stages,
            stage_column=stage_column,
            estimate_shelf_life=estimate_shelf_life,
            shelf_life_threshold=shelf_life_threshold,
            shelf_life_index=shelf_life_index,
        )

        # Provide backwards-compatible key expected by tests
        if "trajectory" not in results:
            results["trajectory"] = results.get("trajectory_models", {})

        # Record metrics
        self.bundle.add_metrics("heating_trajectory", results.get("trajectory_models", {}))
        if "stage_classification" in results:
            self.bundle.add_metrics("stage_classification", results["stage_classification"]["metrics"])
        if "shelf_life" in results:
            self.bundle.add_metrics("shelf_life", results["shelf_life"])

        self.bundle.run_record.add_step(
            "heating_trajectory",
            hashlib.sha256(json.dumps(results.get("trajectory_models", {}), sort_keys=True).encode()).hexdigest()[:8],
            metadata={"time_column": time_column, "indices": indices},
        )
        self._steps_applied.append("heating_trajectory")

        return results
