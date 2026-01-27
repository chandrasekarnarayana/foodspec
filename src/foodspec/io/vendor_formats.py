from __future__ import annotations

"""Vendor format readers (SPC, OPUS).

These readers rely on optional third-party packages. Informative `ImportError`
messages are raised when dependencies are missing.
"""


from pathlib import Path

import numpy as np
import pandas as pd

from foodspec.data_objects.spectra_set import FoodSpectrumSet


def _require(pkg_names: list[str], extra: str):
    for name in pkg_names:
        try:
            return __import__(name)
        except ImportError:
            continue
    raise ImportError(
        f"{extra.upper()} support requires optional packages {pkg_names}. Install with: pip install foodspec[{extra}]"
    )


def read_spc(path: str | Path, modality: str = "raman") -> FoodSpectrumSet:
    """Read an SPC file into `FoodSpectrumSet`.

    Attempts to import known SPC readers (e.g., "spc" or "spc_io"). If none
    are available, an informative `ImportError` is raised.

    Args:
        path: Path to the SPC file.
        modality: Spectroscopy modality label.

    Returns:
        A `FoodSpectrumSet` constructed from the SPC data.

    Raises:
        ImportError: If an SPC reader dependency is not installed.
    """

    spc_mod = _require(["spc", "spc_io"], "spc")
    if hasattr(spc_mod, "File"):
        data = spc_mod.File(str(path))
        wn = data.x
        inten = data.y[np.newaxis, :]
    elif hasattr(spc_mod, "read"):
        data = spc_mod.read(path)
        wn = data.x
        inten = data.y[np.newaxis, :]
    else:  # pragma: no cover
        raise ImportError("SPC reader module did not expose expected API (x/y).")
    metadata = pd.DataFrame({"sample_id": [Path(path).stem]})
    return FoodSpectrumSet(x=inten, wavenumbers=wn, metadata=metadata, modality=modality)


def read_opus(path: str | Path, modality: str = "ftir") -> FoodSpectrumSet:
    """Read a Bruker OPUS file into `FoodSpectrumSet`.

    Uses the optional `brukeropusreader` package when available; raises an
    informative `ImportError` otherwise.

    Args:
        path: Path to the OPUS file.
        modality: Spectroscopy modality label.

    Returns:
        A `FoodSpectrumSet` parsed from the OPUS data.

    Raises:
        ImportError: If `brukeropusreader` is not installed.
    """

    opus_mod = _require(["brukeropusreader"], "opus")
    df = opus_mod.read_file(str(path))
    if "x" in df and "y" in df:
        wn = df["x"].to_numpy()
        inten = df["y"].to_numpy()[np.newaxis, :]
    else:
        wn = df.index.to_numpy()
        inten = df.iloc[:, 0].to_numpy()[np.newaxis, :]
    metadata = pd.DataFrame({"sample_id": [Path(path).stem]})
    return FoodSpectrumSet(x=inten, wavenumbers=wn, metadata=metadata, modality=modality)
