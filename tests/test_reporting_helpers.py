from pathlib import Path

from foodspec.reporting import (
    summarize_preprocessing_pipeline,
    summarize_stats_results,
    summarize_model_performance,
    export_run_metadata,
    export_model_and_metrics,
)


def test_reporting_helpers_summaries(tmp_path: Path):
    prep = summarize_preprocessing_pipeline("Pipeline(ALS -> Norm)")
    assert "pipeline" in prep
    stats = summarize_stats_results({"pvalue": 0.01})
    assert "pvalue" in stats
    metrics = summarize_model_performance({"accuracy": 0.9})
    assert metrics["accuracy"] == 0.9
    out = export_run_metadata(tmp_path / "meta.json", prep, metrics, stats, extra_info={"note": "demo"})
    assert out.exists()


def test_export_model_and_metrics(tmp_path: Path):
    class Dummy:
        pass

    dummy = Dummy()
    # simple pickleable object (e.g., dict) to avoid pickling errors with local classes
    model = {"model": "dummy"}
    paths = export_model_and_metrics(model, {"acc": 0.9}, tmp_path / "model_run")
    assert paths["model"].exists()
    assert paths["metrics"].exists()
