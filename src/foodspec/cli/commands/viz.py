from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import typer

from foodspec.reporting.schema import RunBundle
from foodspec.utils.run_artifacts import get_logger, init_run_dir, write_manifest, write_run_summary
from foodspec.viz import api as viz_api

viz_app = typer.Typer(help="Visualization commands.")


PLOT_REGISTRY: Dict[str, callable] = {
    "overlay": viz_api.plot_raw_processed_overlay,
    "spectra_heatmap": viz_api.plot_spectra_heatmap,
    "correlation": viz_api.plot_correlation_heatmap,
    "pca": viz_api.plot_pca_scatter,
    "umap": viz_api.plot_umap_scatter,
    "confusion": viz_api.plot_confusion_matrix,
    "reliability": viz_api.plot_reliability_diagram,
    "workflow": viz_api.plot_workflow_dag,
    "params": viz_api.plot_parameter_map,
    "lineage": viz_api.plot_data_lineage,
    "badge": viz_api.plot_reproducibility_badge,
    "batch_drift": viz_api.plot_batch_drift,
    "stage_diff": viz_api.plot_stage_difference_spectra,
    "replicate_similarity": viz_api.plot_replicate_similarity,
    "temporal_drift": viz_api.plot_temporal_drift,
    "importance_overlay": viz_api.plot_importance_overlay,
    "marker_bands": viz_api.plot_marker_bands,
    "coefficient_heatmap": viz_api.plot_coefficient_heatmap,
    "feature_stability": viz_api.plot_feature_stability,
    "confidence": viz_api.plot_confidence_map,
    "conformal": viz_api.plot_conformal_set_sizes,
    "coverage_efficiency": viz_api.plot_coverage_efficiency,
    "abstention": viz_api.plot_abstention_distribution,
    "xbar_r": viz_api.plot_xbar_r_chart,
    "xbar_s": viz_api.plot_xbar_s_chart,
    "individuals_mr": viz_api.plot_individuals_mr_chart,
    "cusum": viz_api.plot_cusum_chart,
    "ewma": viz_api.plot_ewma_chart,
    "levey_jennings": viz_api.plot_levey_jennings_chart,
    "probability_plot": viz_api.plot_probability_plot,
    "dendrogram": viz_api.plot_dendrogram,
    "pareto": viz_api.plot_pareto_chart,
    "runs": viz_api.plot_runs_analysis,
}


def _parse_plots(plots: str | None, all_plots: bool) -> List[str]:
    if all_plots or not plots:
        return list(PLOT_REGISTRY.keys())
    names = [p.strip() for p in plots.split(",") if p.strip()]
    return names


@viz_app.command("make")
def make_plots(
    run: Path = typer.Option(..., "--run", "-r", help="Run directory with manifest/run_summary."),
    outdir: Path = typer.Option(Path("viz_runs"), "--outdir", help="Output directory for figures."),
    plots: str | None = typer.Option(None, "--plots", help="Comma-separated plot list."),
    all: bool = typer.Option(False, "--all", help="Generate all plots."),
    seed: int = typer.Option(0, "--seed", help="Random seed for deterministic plots."),
):
    """Generate visualization figures from a run bundle."""
    run_dir = init_run_dir(outdir)
    logger = get_logger(run_dir)
    write_manifest(
        run_dir,
        {"command": "viz.make", "inputs": [run], "plots": plots, "all": all, "seed": seed},
    )

    if not run.exists():
        write_run_summary(run_dir, {"status": "fail", "error": f"Run directory not found: {run}"})
        raise typer.Exit(code=2)

    errors: Dict[str, str] = {}
    generated: List[str] = []
    bundle = RunBundle.from_run_dir(run)
    selected = _parse_plots(plots, all)
    for name in selected:
        fn = PLOT_REGISTRY.get(name)
        if fn is None:
            errors[name] = "unknown_plot"
            continue
        try:
            fn(bundle, outdir=run_dir, name=name, seed=seed)
            generated.append(name)
        except Exception as exc:  # pragma: no cover - defensive
            errors[name] = str(exc)
            logger.error("plot_failed", extra={"plot": name, "error": str(exc)})

    summary = {"status": "success", "generated": generated, "errors": errors}
    if errors:
        summary["status"] = "fail"
        write_run_summary(run_dir, summary)
        raise typer.Exit(code=4)
    write_run_summary(run_dir, summary)
    typer.echo(f"Generated {len(generated)} plots in {run_dir}")


__all__ = ["viz_app"]
