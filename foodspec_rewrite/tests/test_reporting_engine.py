import json
from pathlib import Path

import numpy as np

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.manifest import RunManifest
from foodspec.reporting.engine import ReportingEngine
from foodspec.reporting.modes import ReportMode


def _write_csv(path: Path, rows):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _minimal_manifest(tmp_dir: Path) -> RunManifest:
    data_file = tmp_dir / "data.csv"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text("id,label\n1,a\n")
    return RunManifest.build(
        protocol_snapshot={"version": "2.0.0", "data": {"input": str(data_file), "modality": "raman"}},
        data_path=data_file,
        seed=7,
        artifacts={},
    )


def test_reporting_engine_builds_complete_bundle(tmp_path):
    artifacts = ArtifactRegistry(tmp_path)
    artifacts.ensure_layout()

    manifest = _minimal_manifest(tmp_path)
    manifest.save(artifacts.manifest_path)

    metrics = [
        {"fold_id": 0, "accuracy": 0.9, "macro_f1": 0.88},
        {"fold_id": 1, "accuracy": 0.92, "macro_f1": 0.90},
    ]
    _write_csv(artifacts.metrics_path, metrics)

    qc = [{"check": "snr", "status": "pass"}]
    _write_csv(artifacts.qc_path, qc)

    trust_eval = {"ece": 0.03}
    artifacts.trust_eval_path.write_text(json.dumps(trust_eval))

    engine = ReportingEngine(artifacts, manifest, mode=ReportMode.RESEARCH)
    outputs = engine.build()

    assert outputs.html.exists()
    assert outputs.experiment_card_html.exists()
    assert outputs.experiment_card_json.exists()
    assert outputs.dossier_html.exists()
    assert outputs.archive_path.exists()

    # One run produces a complete report set in a single call
    bundle_contents = outputs.archive_path.read_bytes()
    assert bundle_contents  # archive is non-empty

    card = json.loads(outputs.experiment_card_json.read_text())
    assert card["mode"] == ReportMode.RESEARCH.value
    assert card["headline_metric"]["name"] == "accuracy"

    dossier_html = outputs.dossier_html.read_text()
    assert "Scientific Dossier" in dossier_html
    assert "Results" in dossier_html  # RESEARCH mode includes metrics


def test_run_comparison_dashboard(tmp_path):
    run_dirs = []
    for i in range(2):
        run_dir = tmp_path / f"run_{i}"
        reg = ArtifactRegistry(run_dir)
        reg.ensure_layout()
        manifest = _minimal_manifest(run_dir)
        manifest.save(reg.manifest_path)
        metrics = [{"fold_id": 0, "accuracy": 0.8 + 0.05 * i, "macro_f1": 0.75 + 0.05 * i}]
        _write_csv(reg.metrics_path, metrics)
        run_dirs.append(run_dir)

    engine = ReportingEngine(ArtifactRegistry(tmp_path), {})
    dashboard = engine.compare_runs(run_dirs, output_dir=tmp_path / "comparison")

    assert dashboard.exists()
    assert (dashboard.parent / "leaderboard.csv").exists()
    assert (dashboard.parent / "run_comparison_radar.png").exists()


def test_paper_style_dpi_default(tmp_path):
    from foodspec.viz.paper_styles import apply_paper_style, DEFAULT_DPI
    import matplotlib.pyplot as plt

    params = apply_paper_style("joss", dpi=200)
    assert params["savefig.dpi"] == DEFAULT_DPI
    # ensure it sets rcParams
    fig, ax = plt.subplots()
    ax.plot(np.arange(3), np.arange(3))
    out = tmp_path / "style.png"
    fig.savefig(out)
    assert out.exists()
