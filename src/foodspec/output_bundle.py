from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def create_run_folder(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{ts}_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "figures" / "hsi").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "hsi").mkdir(exist_ok=True)
    return run_dir


def save_report_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def save_report_html(path: Path, text: str):
    html = "<html><body><pre>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>"
    path.write_text(html, encoding="utf-8")


def save_tables(run_dir: Path, tables: Dict[str, object]):
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name, df in tables.items():
        if df is None:
            continue
        out = tables_dir / f"{name}.csv"
        df.to_csv(out, index=False)


def save_figures(run_dir: Path, figures: Dict[str, object]):
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    for name, fig in figures.items():
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            plt = None
            np = None
        # Allow nested names like "hsi/labels"
        target_path = figs_dir / f"{name}.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if plt is not None and np is not None and isinstance(fig, np.ndarray):
            plt.figure()
            plt.imshow(fig, cmap="viridis")
            plt.axis("off")
            plt.savefig(target_path, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            fig.savefig(target_path, dpi=200)


def save_metadata(run_dir: Path, meta: Dict):
    meta_path = run_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def append_log(run_dir: Path, message: str):
    log_path = run_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.strip() + "\n")


def save_index(run_dir: Path, metadata: Dict, tables: Dict[str, object], figures: Dict[str, object], warnings: List[str]):
    """
    Lightweight index.json for quick inspection of a run.
    Lists tables, figures, metadata, and warnings/notes.
    """
    index = {
        "run_id": run_dir.name,
        "metadata": metadata,
        "tables": list(tables.keys()),
        "figures": list(figures.keys()),
        "warnings": warnings,
        "models": metadata.get("models", []),
        "validation": metadata.get("validation_strategy"),
        "harmonization": metadata.get("harmonization", {}),
    }
    (run_dir / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
