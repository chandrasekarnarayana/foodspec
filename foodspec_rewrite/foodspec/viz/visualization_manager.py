"""
Visualization Manager

Orchestrates visualization generation from protocol and manifest definitions.
- Reads protocol/manifest (dicts or JSON/YAML files)
- Determines which plots to generate
- Dispatches to the appropriate plot functions
- Handles save paths and logging
- Returns a summary of generated figures

Public entry point: run_all_visualizations(run_context)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt

from foodspec.viz import (
    plot_embedding,
    plot_embedding_comparison,
    plot_preprocessing_comparison,
    plot_processing_stages,
    plot_pipeline_dag,
    plot_batch_drift,
    plot_importance_overlay,
    plot_confidence_map,
)

try:  # Optional YAML support if PyYAML is available
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when yaml is unavailable
    yaml = None


PlotParams = Dict[str, Any]
MappingLike = Union[Dict[str, Any], str, Path, None]


PLOT_DISPATCH = {
    "embedding": plot_embedding,
    "embedding_comparison": plot_embedding_comparison,
    "processing_stages": plot_processing_stages,
    "preprocessing_comparison": plot_preprocessing_comparison,
    "pipeline_dag": plot_pipeline_dag,
    "batch_drift": plot_batch_drift,
    "importance_overlay": plot_importance_overlay,
    "confidence_map": plot_confidence_map,
}


@dataclass
class VisualizationResult:
    """Represents the outcome of a visualization task."""

    name: str
    plot_type: str
    save_path: Optional[Path]
    status: str  # success | failed | skipped
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "plot_type": self.plot_type,
            "save_path": str(self.save_path) if self.save_path else None,
            "status": self.status,
            "error": self.error,
        }


class VisualizationManager:
    """Determines and executes visualization tasks defined in protocol/manifest."""

    def __init__(
        self,
        protocol: MappingLike,
        manifest: MappingLike,
        data_store: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path, None] = None,
        logger: Optional[logging.Logger] = None,
        default_dpi: int = 300,
    ) -> None:
        self.protocol = self._load_mapping(protocol)
        self.manifest = self._load_mapping(manifest)
        self.data_store = data_store or {}
        self.output_dir = Path(output_dir or "outputs/visualizations")
        self.default_dpi = default_dpi
        self.logger = logger or logging.getLogger("foodspec.visualization_manager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def run_all(self) -> List[VisualizationResult]:
        tasks = self._extract_tasks()
        results: List[VisualizationResult] = []

        if not tasks:
            self.logger.info("No visualization tasks found in protocol/manifest")
            return results

        for task in tasks:
            results.append(self._run_task(task))

        return results

    def _load_mapping(self, source: MappingLike) -> Dict[str, Any]:
        if source is None:
            return {}
        if isinstance(source, dict):
            return source
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Cannot load mapping from missing file: {path}")
        if path.suffix.lower() in {".json"}:
            return json.loads(path.read_text())
        if path.suffix.lower() in {".yml", ".yaml"} and yaml is not None:
            return yaml.safe_load(path.read_text())
        # Fallback: try JSON first, then YAML if available
        try:
            return json.loads(path.read_text())
        except Exception:
            if yaml is not None:
                return yaml.safe_load(path.read_text())
            raise ValueError(f"Unsupported mapping format for {path}")

    def _extract_tasks(self) -> List[Dict[str, Any]]:
        visualizations: Iterable[Any] = (
            self.protocol.get("visualizations")
            or self.manifest.get("visualizations")
            or []
        )
        tasks: List[Dict[str, Any]] = []
        for idx, entry in enumerate(visualizations):
            tasks.append(self._normalize_task(entry, idx))
        return tasks

    def _normalize_task(self, entry: Any, idx: int) -> Dict[str, Any]:
        if isinstance(entry, str):
            return {
                "name": entry,
                "type": entry,
                "params": {},
                "data_keys": {},
            }
        if isinstance(entry, dict):
            plot_type = entry.get("type") or entry.get("name") or f"plot_{idx}"
            return {
                "name": entry.get("name") or plot_type,
                "type": plot_type,
                "params": entry.get("params", {}),
                "data_keys": entry.get("data_keys", {}),
                "save_path": entry.get("save_path") or entry.get("save_as"),
            }
        # Unknown entry type; mark as skipped
        return {
            "name": f"plot_{idx}",
            "type": "unknown",
            "params": {},
            "data_keys": {},
        }

    def _resolve_save_path(self, task: Dict[str, Any]) -> Path:
        name = task.get("name") or task.get("type") or "figure"
        provided = task.get("save_path") or task.get("save_as")
        path = Path(provided) if provided else Path(f"{name}.png")
        if not path.suffix:
            path = path.with_suffix(".png")
        if not path.is_absolute():
            path = self.output_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _prepare_params(self, task: Dict[str, Any]) -> PlotParams:
        params: PlotParams = dict(task.get("params", {}))
        data_keys = task.get("data_keys", {}) or {}
        for param_key, store_key in data_keys.items():
            if param_key in params:
                continue
            if isinstance(store_key, str) and store_key in self.data_store:
                params[param_key] = self.data_store[store_key]
        return params

    def _run_task(self, task: Dict[str, Any]) -> VisualizationResult:
        plot_type = task.get("type", "unknown")
        name = task.get("name") or plot_type
        func = PLOT_DISPATCH.get(plot_type)

        if func is None:
            self.logger.warning("Skipping plot '%s': unsupported type", plot_type)
            return VisualizationResult(
                name=name,
                plot_type=plot_type,
                save_path=None,
                status="skipped",
                error=f"Unsupported plot type: {plot_type}",
            )

        params = self._prepare_params(task)
        save_path = params.get("save_path")
        if save_path is None:
            save_path = self._resolve_save_path(task)
            params["save_path"] = save_path
        params.setdefault("dpi", self.default_dpi)

        status = "success"
        error: Optional[str] = None
        fig = None
        try:
            fig = func(**params)
            self.logger.info("Generated plot '%s' (%s) -> %s", name, plot_type, save_path)
        except Exception as exc:  # pragma: no cover - guarded by logging
            status = "failed"
            error = str(exc)
            self.logger.exception("Failed to generate plot '%s' (%s)", name, plot_type)
            save_path = None
        finally:
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:
                    pass

        return VisualizationResult(
            name=name,
            plot_type=plot_type,
            save_path=save_path,
            status=status,
            error=error,
        )


def run_all_visualizations(run_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convenience wrapper to execute all visualizations.

    Parameters
    ----------
    run_context : dict
        Expected keys:
        - protocol: dict or path to JSON/YAML
        - manifest: dict or path to JSON/YAML
        - data or data_store: mapping of data arrays keyed by string
        - output_dir or save_dir: optional base directory for outputs
        - logger: optional logging.Logger

    Returns
    -------
    list of dict
        Summary per visualization with keys: name, plot_type, save_path, status, error
    """
    if run_context is None:
        raise ValueError("run_context is required")

    protocol = run_context.get("protocol")
    manifest = run_context.get("manifest")
    data_store = run_context.get("data") or run_context.get("data_store") or {}
    output_dir = run_context.get("output_dir") or run_context.get("save_dir")
    logger = run_context.get("logger")

    manager = VisualizationManager(
        protocol=protocol,
        manifest=manifest,
        data_store=data_store,
        output_dir=output_dir,
        logger=logger,
    )
    results = manager.run_all()
    return [result.to_dict() for result in results]
