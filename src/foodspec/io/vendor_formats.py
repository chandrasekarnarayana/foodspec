"""
Vendor format readers (SPC, OPUS). Optional dependencies are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet


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
    """
    Read an SPC file into FoodSpectrumSet (optional dependency).

    Tries to import known SPC readers (e.g., `spc` or `spc_io`). Raises an informative
    ImportError if dependencies are missing.
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
    """
    Read a Bruker OPUS file into FoodSpectrumSet (optional dependency).

    Uses `brukeropusreader` if available; raises informative ImportError otherwise.
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
