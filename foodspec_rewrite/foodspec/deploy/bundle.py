"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

Model bundle export and import for deployment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import numpy as np


def _get_package_versions() -> Dict[str, str]:
    """Collect versions of key dependencies."""
    versions = {"python": sys.version.split()[0]}
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "pydantic"]:
        try:
            mod = __import__(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:  # pragma: no cover
            versions[pkg] = "not installed"
    return versions


def save_bundle(
    run_dir: Path | str,
    bundle_dir: Path | str,
    protocol: Mapping[str, Any],
    preprocess_pipeline: Optional[List[Dict[str, Any]]] = None,
    model_path: Optional[Path | str] = None,
    label_encoder: Optional[Dict[int, str]] = None,
    x_grid: Optional[np.ndarray] = None,
    metadata_schema: Optional[Dict[str, str]] = None,
) -> Path:
    """Export a deployment bundle with all necessary artifacts.

    Parameters
    ----------
    run_dir : Path | str
        Source run directory (unused but included for consistency).
    bundle_dir : Path | str
        Target directory for the bundle.
    protocol : Mapping[str, Any]
        Expanded protocol snapshot (from manifest or ProtocolV2).
    preprocess_pipeline : List[Dict[str, Any]], optional
        Preprocessing pipeline specification (list of step dicts).
    model_path : Path | str, optional
        Path to trained model file (will be copied into bundle).
    label_encoder : Dict[int, str], optional
        Mapping from integer labels to class names.
    x_grid : ndarray, optional
        Feature grid (e.g., wavenumbers or feature names).
    metadata_schema : Dict[str, str], optional
        Schema mapping canonical keys to expected column types.

    Returns
    -------
    bundle_path : Path
        Path to the created bundle directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> import numpy as np
    >>> protocol = {"version": "2.0.0", "data": {"modality": "raman"}}
    >>> bundle_path = save_bundle(
    ...     run_dir=Path("/tmp/run"),
    ...     bundle_dir=Path("/tmp/bundle"),
    ...     protocol=protocol,
    ...     x_grid=np.array([400, 500, 600]),
    ... )  # doctest: +SKIP
    """

    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    # Save protocol
    protocol_file = bundle_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol, indent=2, default=str))

    # Save preprocessing pipeline
    if preprocess_pipeline is not None:
        pipeline_file = bundle_path / "preprocess_pipeline.json"
        pipeline_file.write_text(json.dumps(preprocess_pipeline, indent=2))

    # Copy model
    if model_path is not None:
        model_src = Path(model_path)
        if model_src.exists():
            model_dst = bundle_path / "model.joblib"
            import shutil

            shutil.copy2(model_src, model_dst)

    # Save label encoder
    if label_encoder is not None:
        encoder_file = bundle_path / "label_encoder.json"
        encoder_file.write_text(json.dumps(label_encoder, indent=2))

    # Save x_grid
    if x_grid is not None:
        grid_file = bundle_path / "x_grid.npy"
        np.save(grid_file, x_grid)

    # Save metadata schema
    if metadata_schema is not None:
        schema_file = bundle_path / "metadata_schema.json"
        schema_file.write_text(json.dumps(metadata_schema, indent=2))

    # Save package versions
    versions_file = bundle_path / "package_versions.json"
    versions_file.write_text(json.dumps(_get_package_versions(), indent=2))

    # Create bundle manifest
    manifest = {
        "bundle_format_version": "1.0.0",
        "contents": {
            "protocol": protocol_file.name,
            "preprocess_pipeline": "preprocess_pipeline.json" if preprocess_pipeline else None,
            "model": "model.joblib" if model_path else None,
            "label_encoder": "label_encoder.json" if label_encoder else None,
            "x_grid": "x_grid.npy" if x_grid is not None else None,
            "metadata_schema": "metadata_schema.json" if metadata_schema else None,
            "package_versions": versions_file.name,
        },
    }
    manifest_file = bundle_path / "bundle_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))

    return bundle_path


def load_bundle(bundle_dir: Path | str) -> Dict[str, Any]:
    """Load a deployment bundle.

    Parameters
    ----------
    bundle_dir : Path | str
        Path to the bundle directory.

    Returns
    -------
    bundle : dict
        Dictionary with keys: protocol, preprocess_pipeline, model, label_encoder,
        x_grid, metadata_schema, package_versions, manifest.

    Raises
    ------
    FileNotFoundError
        If bundle manifest is missing.
    ValueError
        If bundle format is unsupported.

    Examples
    --------
    >>> from pathlib import Path
    >>> bundle = load_bundle(Path("/tmp/bundle"))  # doctest: +SKIP
    >>> bundle["protocol"]["version"]  # doctest: +SKIP
    '2.0.0'
    """

    bundle_path = Path(bundle_dir)
    manifest_file = bundle_path / "bundle_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Bundle manifest not found at {manifest_file}")

    manifest = json.loads(manifest_file.read_text())
    if manifest.get("bundle_format_version") != "1.0.0":
        raise ValueError(f"Unsupported bundle format: {manifest.get('bundle_format_version')}")

    contents = manifest["contents"]
    bundle: Dict[str, Any] = {"manifest": manifest}

    # Load protocol
    protocol_file = bundle_path / contents["protocol"]
    bundle["protocol"] = json.loads(protocol_file.read_text())

    # Load preprocess pipeline
    if contents.get("preprocess_pipeline"):
        pipeline_file = bundle_path / contents["preprocess_pipeline"]
        bundle["preprocess_pipeline"] = json.loads(pipeline_file.read_text())
    else:
        bundle["preprocess_pipeline"] = None

    # Load model
    if contents.get("model"):
        model_file = bundle_path / contents["model"]
        bundle["model"] = joblib.load(model_file)
    else:
        bundle["model"] = None

    # Load label encoder
    if contents.get("label_encoder"):
        encoder_file = bundle_path / contents["label_encoder"]
        encoder_data = json.loads(encoder_file.read_text())
        # Convert string keys back to int
        bundle["label_encoder"] = {int(k): v for k, v in encoder_data.items()}
    else:
        bundle["label_encoder"] = None

    # Load x_grid
    if contents.get("x_grid"):
        grid_file = bundle_path / contents["x_grid"]
        bundle["x_grid"] = np.load(grid_file)
    else:
        bundle["x_grid"] = None

    # Load metadata schema
    if contents.get("metadata_schema"):
        schema_file = bundle_path / contents["metadata_schema"]
        bundle["metadata_schema"] = json.loads(schema_file.read_text())
    else:
        bundle["metadata_schema"] = None

    # Load package versions
    versions_file = bundle_path / contents["package_versions"]
    bundle["package_versions"] = json.loads(versions_file.read_text())

    return bundle


__all__ = ["save_bundle", "load_bundle"]
