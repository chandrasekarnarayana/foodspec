from __future__ import annotations

"""Standardized figure saving utilities with metadata sidecars."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from foodspec._version import __version__
from foodspec.utils.run_artifacts import safe_json_dump


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(metadata)
    payload.setdefault("code_version", __version__)
    payload.setdefault("timestamp", _utc_now())
    return payload


def save_figure(
    fig,
    outpath: Path | str,
    metadata: Mapping[str, Any],
    *,
    fmt: Iterable[str] = ("png", "svg"),
    dpi: int = 300,
) -> list[Path]:
    """Save a figure to multiple formats with a JSON metadata sidecar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    outpath : Path | str
        Base output path (no extension recommended).
    metadata : Mapping[str, Any]
        Metadata payload for sidecar. Must include description/inputs.
    fmt : iterable of str
        Output formats (e.g., ("png", "svg")).
    dpi : int
        DPI for raster formats.
    """
    base = Path(outpath)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for ext in fmt:
        target = base.with_suffix(f".{ext}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        outputs.append(target)

    meta_payload = _normalize_metadata(metadata)
    meta_path = base.with_suffix(".meta.json")
    safe_json_dump(meta_path, meta_payload)
    return outputs


__all__ = ["save_figure"]
