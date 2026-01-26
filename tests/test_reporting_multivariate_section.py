from pathlib import Path

import pandas as pd
import pytest

from foodspec.report.sections.multivariate import build_multivariate_section


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "multivariate" / "pca").mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(6)],
        "pc1": [1, 2, 1.5, -0.5, -1.2, 2.4],
        "pc2": [0.4, -0.2, 0.9, -1.1, 0.2, 1.3],
        "label": ["a", "a", "b", "b", "a", "b"],
        "batch": ["b1", "b1", "b2", "b2", "b1", "b2"],
    })
    loadings = pd.DataFrame({
        "component": ["pc1", "pc2"],
        "f1": [0.6, 0.2],
        "f2": [-0.4, 0.1],
        "f3": [0.1, -0.3],
    })
    scores.to_csv(run_dir / "multivariate" / "pca" / "scores.csv", index=False)
    loadings.to_csv(run_dir / "multivariate" / "pca" / "loadings.csv", index=False)
        (run_dir / "multivariate" / "pca" / "summary.json").write_text("{\"explained_variance\": [0.7, 0.2]}")
    (run_dir / "qc").mkdir(parents=True, exist_ok=True)
    (run_dir / "qc" / "qc_summary.json").write_text("{\"multivariate\": {\"outliers\": {\"n_flagged\": 1}}}")
    return run_dir


class DummyContext:
    def __init__(self, run_dir: Path, trust_outputs: dict):
        self.run_dir = run_dir
        self.trust_outputs = trust_outputs


def test_multivariate_section_builds_figures(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    trust_outputs = {"qc_summary": {"multivariate": {"outliers": {"n_flagged": 1}}}}
    section = build_multivariate_section(DummyContext(run_dir, trust_outputs))

    assert section["methods"], "Should detect at least one method"
    pca_entry = section["methods"][0]
    fig_paths = pca_entry["figures"]
    assert any("scores_label" in str(p) for p in fig_paths)
    assert any((run_dir / p).exists() for p in fig_paths)


def test_template_fragment_includes_multivariate(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    trust_outputs = {"qc_summary": {"multivariate": {"outliers": {"n_flagged": 1}}}}
    section = build_multivariate_section(DummyContext(run_dir, trust_outputs))

    # Minimal context to render multivariate portion
    from jinja2 import Environment, FileSystemLoader

    template_dir = Path(__file__).resolve().parents[1] / "src" / "foodspec" / "reporting" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("base.html")

    manifest_stub = {
        "seed": 0,
        "protocol_hash": "deadbeef" * 4,
        "data_fingerprint": "feedface" * 4,
        "start_time": "",
        "end_time": "",
        "duration_seconds": 0.0,
        "python_version": "",
        "platform": "",
    }
    context_payload = {
        "title": "Test Report",
        "mode": "research",
        "mode_description": "",
        "enabled_sections": ["multivariate"],
        "manifest": manifest_stub,
        "protocol": {},
        "metrics": [],
        "predictions": [],
        "qc": [],
        "trust_outputs": trust_outputs,
        "figures": {},
        "multivariate": {},
        "available_artifacts": [],
        "multivariate_section": section,
    }

    html = template.render(context_payload)
    assert "Multivariate Analysis" in html
    assert "PCA" in html or "pca" in html.lower()
